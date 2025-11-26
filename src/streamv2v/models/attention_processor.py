from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.import_utils import is_xformers_available

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class MultiStateAttentionProcessor:
    def __init__(self, name=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("MultiStateAttentionProcessor requires PyTorch 2.0.")
        self.name = name
        self.last_key = None
        self.last_value = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        discrepancy_mask: Optional[torch.FloatTensor] = None,
        bank_kvs_by_layer: Optional[dict] = None,
        prev_kvs_by_layer: Optional[dict] = None,
    ) -> torch.FloatTensor:
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        original_input_batch_size = hidden_states.shape[0]
        original_input_ndim = hidden_states.ndim

        if original_input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = original_input_batch_size
        
        sequence_length = hidden_states.shape[1]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_selfattn = encoder_hidden_states is None
        if is_selfattn:
            encoder_hidden_states_processed = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states_processed = attn.norm_encoder_hidden_states(encoder_hidden_states)
        else:
            encoder_hidden_states_processed = encoder_hidden_states

        key_curr = attn.to_k(encoder_hidden_states_processed)
        value_curr = attn.to_v(encoder_hidden_states_processed)

        if is_selfattn:
            self.last_key = key_curr.clone()
            self.last_value = value_curr.clone()

        key_bank, value_bank = None, None
        if bank_kvs_by_layer and self.name in bank_kvs_by_layer:
            key_bank, value_bank = bank_kvs_by_layer[self.name]
            key_bank = key_bank.to(query.device, dtype=query.dtype)
            value_bank = value_bank.to(query.device, dtype=query.dtype)

        key_prev_warped, value_prev_warped = None, None
        if prev_kvs_by_layer and self.name in prev_kvs_by_layer:
            key_prev_warped, value_prev_warped = prev_kvs_by_layer[self.name]
            key_prev_warped = key_prev_warped.to(query.device, dtype=query.dtype)
            value_prev_warped = value_prev_warped.to(query.device, dtype=query.dtype)
        
        key_list = [key_curr]
        value_list = [value_curr]
        
        attn_bias = None
        if is_selfattn and discrepancy_mask is not None:
            # If CFG is enabled, hidden_states batch size is doubled.
            # The mask, based on single latents, needs to be duplicated to match.
            if hidden_states.shape[0] != discrepancy_mask.shape[0]:
                discrepancy_mask = discrepancy_mask.repeat(hidden_states.shape[0] // discrepancy_mask.shape[0], 1, 1, 1)

            # Prepare bias based on discrepancy mask (M_diff)
            # M_diff is high (close to 1) for conflict, low (close to 0) for consistency
            
            # (1 - M_diff) is high for consistency, low for conflict
            consistency_mask = 1.0 - discrepancy_mask
            bias_mask_consistency_flat = F.interpolate(consistency_mask, size=(1, sequence_length), mode='bilinear', align_corners=False)
            bias_mask_conflict_flat = F.interpolate(discrepancy_mask, size=(1, sequence_length), mode='bilinear', align_corners=False)
            attn_bias = torch.zeros(batch_size, attn.heads, sequence_length, 0, device=query.device, dtype=query.dtype)
            bias_curr = torch.zeros(batch_size, attn.heads, sequence_length, key_curr.shape[1], device=query.device, dtype=query.dtype)
            attn_bias = torch.cat([attn_bias, bias_curr], dim=-1)

            if key_prev_warped is not None:
                key_list.append(key_prev_warped)
                value_list.append(value_prev_warped)
                bias_prev_reshaped = bias_mask_consistency_flat.permute(0, 1, 3, 2).contiguous()
                bias_prev_expanded = (1.0 - bias_prev_reshaped) * -1e4
                bias_prev_expanded = bias_prev_expanded.repeat(1, attn.heads, 1, key_prev_warped.shape[1])
                attn_bias = torch.cat([attn_bias, bias_prev_expanded], dim=-1)

            if key_bank is not None:
                key_list.append(key_bank)
                value_list.append(value_bank)
                bias_bank_reshaped = bias_mask_conflict_flat.permute(0, 1, 3, 2).contiguous()
                bias_bank_expanded = (1.0 - bias_bank_reshaped) * -1e4
                bias_bank_expanded = bias_bank_expanded.repeat(1, attn.heads, 1, key_bank.shape[1])
                attn_bias = torch.cat([attn_bias, bias_bank_expanded], dim=-1)
        else:
            attn_bias = attention_mask
        
        key = torch.cat(key_list, dim=1)
        value = torch.cat(value_list, dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if original_input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(original_input_batch_size, channel, height, width)
        
        return hidden_states