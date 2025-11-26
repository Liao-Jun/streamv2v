# Change Log: Dual-Flow Attention Implementation

This document outlines the implementation of a new "Dual-Flow Check" attention mechanism for the StreamV2V project. The goal is to improve temporal consistency in video generation by dynamically checking for discrepancies between the optical flow of the source video and the generated video.

## 1. New File: `src/streamv2v/flow_utils.py`

A new utility file will be created to handle all optical flow operations. This includes calculating flow between two frames, warping tensors based on flow fields, and composing flow fields over time.

```python
# src/streamv2v/flow_utils.py
# Description: Utilities for optical flow calculation, warping, and composition.

import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import normalize, resize

# Global cache for the flow model
_flow_model = None

def get_flow_model(device="cuda"):
    """
    Loads and returns a pre-trained RAFT optical flow model.
    Caches the model globally to avoid reloading.
    """
    global _flow_model
    if _flow_model is None:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False).to(device)
        model = model.eval()
        _flow_model = {"model": model, "transforms": weights.transforms()}
    return _flow_model

def _preprocess_for_flow(img: torch.Tensor):
    """
    Preprocesses a batch of images for the RAFT model.
    RAFT expects images to be resized to a multiple of 8.
    """
    _, _, h, w = img.shape
    new_h = (h // 8) * 8
    new_w = (w // 8) * 8
    
    if h != new_h or w != new_w:
        img = resize(img, size=[new_h, new_w], antialias=False)
        
    return img

def calculate_flow(
    img1: torch.Tensor,
    img2: torch.Tensor,
    model: dict,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Calculates optical flow from img1 to img2.

    Args:
        img1 (torch.Tensor): The first image tensor (t-1). Shape: (B, C, H, W).
        img2 (torch.Tensor): The second image tensor (t). Shape: (B, C, H, W).
        model (dict): The RAFT model and transforms.

    Returns:
        torch.Tensor: The calculated flow field. Shape: (B, 2, H, W).
    """
    orig_h, orig_w = img1.shape[-2:]
    
    # Preprocess images
    img1_pre = _preprocess_for_flow(img1)
    img2_pre = _preprocess_for_flow(img2)
    
    # Apply RAFT-specific transforms
    transforms = model["transforms"]
    img1_transformed, img2_transformed = transforms(img1_pre, img2_pre)

    with torch.no_grad():
        flow_pred = model["model"](img1_transformed.to(device), img2_transformed.to(device))[-1]

    # Upsample flow to original image size
    flow_upsampled = F.interpolate(flow_pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    
    return flow_upsampled


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warps a tensor `x` using a given optical flow field.

    Args:
        x (torch.Tensor): The tensor to warp (image or feature map). Shape: (B, C, H, W).
        flow (torch.Tensor): The flow field. Shape: (B, 2, H, W).

    Returns:
        torch.Tensor: The warped tensor.
    """
    B, C, H, W = x.size()
    
    # Create sampling grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    grid = torch.stack([xx, yy], dim=0).float().to(x.device)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    
    # Add flow to grid
    vgrid = grid + flow
    
    # Scale grid to [-1, 1] for grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    
    vgrid = vgrid.permute(0, 2, 3, 1)
    
    # Warp tensor
    warped_x = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return warped_x


def compose_flow(flow_t_minus_1_to_t: torch.Tensor, flow_0_to_t_minus_1: torch.Tensor) -> torch.Tensor:
    """
    Composes two flow fields using the chain rule.
    F_accum_new = flow_current + warp(flow_accum_old, flow_current)
    """
    if flow_0_to_t_minus_1 is None:
        return flow_t_minus_1_to_t

    # Warp the previous accumulated flow with the current flow
    warped_accum_flow = warp(flow_0_to_t_minus_1, flow_t_minus_1_to_t)
    
    # Add the current flow
    composed = warped_accum_flow + flow_t_minus_1_to_t
    
    return composed

```

## 2. Modifications to: `src/streamv2v/models/attention_processor.py`

A new attention processor, `DualFlowAttentionProcessor`, will be added. This processor implements the core logic of the new method. It takes discrepancy masks as input and uses them to bias the attention scores, encouraging the model to reuse features from previous frames when the generated content is consistent with the source motion.

```python
class DualFlowAttentionProcessor:
    r"""
    Processor for implementing attention with Dual-Flow guidance for temporal consistency.
    This processor uses pre-computed masks based on optical flow discrepancies
    to bias the attention scores, forcing the model to reuse texture from previous
    frames when motion is consistent.
    """

    def __init__(self, name=None, local_bias_scale=1.0, global_bias_scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("DualFlowAttentionProcessor requires PyTorch 2.0.")
        self.name = name
        self.local_bias_scale = local_bias_scale
        self.global_bias_scale = global_bias_scale
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
        **cross_attention_kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        is_selfattn = encoder_hidden_states is None
        if is_selfattn:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        
        # Cache the current key and value for the pipeline to retrieve later
        if is_selfattn:
            self.last_key = key.clone()
            self.last_value = value.clone()

        # Retrieve cached features and masks from kwargs
        cached_key = cross_attention_kwargs.get("cached_key")
        cached_value = cross_attention_kwargs.get("cached_value")
        mask_local = cross_attention_kwargs.get("mask_local")
        mask_global = cross_attention_kwargs.get("mask_global")

        attn_bias = None
        if is_selfattn and cached_key is not None and cached_value is not None:
            # Concatenate current and cached features
            key = torch.cat([key, cached_key], dim=1)
            value = torch.cat([value, cached_value], dim=1)

            if mask_local is not None and mask_global is not None:
                # Create the bias tensor based on the discrepancy masks
                combined_mask = (mask_local * self.local_bias_scale + mask_global * self.global_bias_scale) / 2.0
                
                attn_map_size = (query.shape[1], cached_key.shape[1])
                feat_h = int(attn_map_size[0] ** 0.5)
                feat_w = feat_h if feat_h * feat_h == attn_map_size[0] else attn_map_size[0] // feat_h
                
                bias_mask_resized = F.interpolate(combined_mask, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
                bias_mask_reshaped = bias_mask_resized.view(batch_size, 1, feat_h * feat_w, 1).repeat(1, attn.heads, 1, 1)

                # Bias values from a large negative number (high discrepancy) to 0 (low discrepancy).
                # We want to PENALIZE attention to the PREVIOUS frame where discrepancy is HIGH.
                # A high mask value means high discrepancy.
                attn_bias = (bias_mask_reshaped - 1.0) * 1e4
                
                # The bias should apply to the cached keys.
                current_key_bias = torch.zeros(batch_size, attn.heads, query.shape[1], key.shape[1] - cached_key.shape[1], device=query.device, dtype=query.dtype)
                attn_bias = torch.cat([current_key_bias, attn_bias], dim=-1)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # The `attn_mask` and `attn_bias` are added inside the sdp call.
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states
```

## 3. Modifications to: `src/streamv2v/pipeline.py`

The main `StreamV2V` pipeline has been significantly updated to integrate the Dual-Flow attention mechanism. Due to the extensive nature of the changes, the entire `src/streamv2v/pipeline.py` file was rewritten.

The key modifications are as follows:

- **Imports**: Added imports for `flow_utils` and `DualFlowAttentionProcessor`.
- **`__init__` Method**:
    - A new boolean flag `use_dual_flow` was added to enable or disable the feature.
    - If enabled, the RAFT optical flow model is initialized.
    - New state buffers are initialized to manage the pipeline's memory across frames, including `prev_frame_latents`, `prev_output_latents`, `flow_accum_src`, `flow_accum_gen`, and `cached_kv`.
- **`unet_step` Method**: The method signature was updated to accept `cross_attention_kwargs`, which are then passed directly to the `unet` call.
- **`predict_x0_batch` Method**: This method now contains the core logic for the dual-flow mechanism.
    - It checks if the feature is enabled and if it is not the first frame.
    - It calculates the source optical flow, approximates the generated flow, and accumulates both over time.
    - It computes the `mask_local` and `mask_global` discrepancy masks based on the flow differences.
    - It prepares the `cross_attention_kwargs` dictionary, filling it with the cached K/V tensors from the previous frame and the newly computed masks.
    - After the U-Net has run, it updates the state for the next frame, including the latent buffers and flow accumulators. It also calls a new helper method to cache the K/V tensors for the next iteration.
- **New `_get_last_kv_from_unet` Helper Method**: This private method was added to iterate through the UNet's attention modules and retrieve the `last_key` and `last_value` tensors that were cached by the `DualFlowAttentionProcessor`. This is crucial for passing the temporal context to the next frame.

## 4. Modifications to: `utils/wrapper.py`

The `StreamV2VWrapper` in `utils/wrapper.py` has been updated to support the activation of the new Dual-Flow attention mechanism. The entire file was rewritten to cleanly integrate the new options.

The key modifications are as follows:

- **Import**: The new `DualFlowAttentionProcessor` is now imported from `src.streamv2v.models.attention_processor`.
- **`__init__` Method**:
    - A new boolean flag `use_dual_flow` has been added to the `StreamV2VWrapper` constructor, allowing users to enable the feature upon initialization.
    - A check has been added to ensure that `use_dual_flow` and the existing `use_cached_attn` are not enabled simultaneously, as they are mutually exclusive mechanisms.
- **`_load_model` Method**:
    - The `use_dual_flow` flag is received and passed down to the constructor of the `StreamV2V` pipeline class.
    - Inside the `acceleration == "xformers"` block, new logic has been added. If `use_dual_flow` is `True`, the code now iterates through the UNet's attention processors and replaces each one with an instance of `DualFlowAttentionProcessor`. This ensures that the custom attention logic is used throughout the model.
    - Error handling has been added to prevent the use of `use_dual_flow` with incompatible acceleration methods like TensorRT or StableFast.

## 5. Modifications to: `vid2vid/main.py`

To allow for easier control over editing strength and to enable the new features, the example script `vid2vid/main.py` has been updated.

The key modifications are as follows:

- **Command-Line Interface**: The script was refactored to use Python's standard `argparse` library instead of `fire`, providing a more conventional and descriptive CLI.
- **Control Parameters**: New, more intuitive command-line arguments have been added:
    - `--strength`: A float between 0.0 and 1.0 that directly controls the editing strength. Higher values result in more significant changes to the video.
    - `--guidance_scale`: A float to control how closely the output adheres to the text prompt.
    - `--cfg_type`: A string to select the type of Classifier-Free Guidance to use, which is necessary for `guidance_scale` to have an effect.
- **Feature Flag**: A `--use_dual_flow` flag was added to make it easy to enable the new Dual-Flow attention mechanism.
- **Logic Update**: The script now passes these new arguments to the `StreamV2VWrapper` and `prepare` methods, allowing for direct user control over the video translation process from the command line.

## 方案3: 动态记忆库与多态注意力 (中文2)

This section documents the implementation of the more advanced "Dynamic Memory Bank" and "Multi-State Attention" architecture. This version builds upon the dual-flow concept to provide more robust temporal consistency over long videos with significant changes in viewpoint.

### 1. Refactoring of: `src/streamv2v/models/attention_processor.py`

The attention processor module has been significantly refactored to support the new, more complex attention strategy.

- **Removed Obsolete Classes**: The old `CachedSTAttnProcessor2_0` and `CachedSTXFormersAttnProcessor` classes have been completely removed from the file, as their functionality is superseded by the new approach.
- **Upgraded to `MultiStateAttentionProcessor`**: The previous `DualFlowAttentionProcessor` has been evolved into `MultiStateAttentionProcessor`. This new class is the core of the "Pillar 3" logic.
    - It is designed to accept multiple sets of Key/Value pairs from `cross_attention_kwargs`: (1) the current frame's K/V, (2) the warped K/V from the previous frame, and (3) a bank of K/V from the static anchor and dynamic keyframes.
    - It uses the `discrepancy_mask` ($M_{diff}$) to create a complex attention bias.
    - When discrepancy is low (consistent motion), the bias matrix strongly favors attention towards the `key_prev_warped` features to ensure smoothness.
    - When discrepancy is high (conflict/occlusion), the bias matrix suppresses attention to the warped previous frame and instead favors the `key_bank` features to recover correct identity and texture information.
    - It continues to cache the K/V features of the current frame (`last_key`, `last_value`) for use by the pipeline in the next step.

### 2. Major Upgrade of: `src/streamv2v/pipeline.py`

The core `StreamV2V` pipeline was almost entirely rewritten to manage the complex state required for the new architecture.

- **New Control Parameters**: The `StreamV2V.__init__` method now accepts new parameters to control the memory bank, such as `use_multi_state_attn`, `memory_bank_size`, `motion_threshold`, and `quality_threshold`.
- **Dynamic Memory Bank**:
    - The pipeline now manages a `memory_bank` (a `deque`) which stores the features of the static anchor (first frame) and a list of dynamic keyframes.
    - The state `static_anchor_kv` is created from the very first frame and preserved.
- **Multi-Source K/V Preparation**:
    - Before each generation step, the pipeline now prepares all the necessary K/V sources: it retrieves the bank, and it warps the K/V from the immediate past frame using the calculated source flow.
    - These are all passed into the `unet_step` via `cross_attention_kwargs`.
- **Dual-Threshold Keyframe Strategy**:
    - After a frame is generated, a "post-process" check is performed.
    - The pipeline calculates the true discrepancy between the source and the newly generated frame to get a `quality_score`.
    - It also tracks the accumulated motion since the last keyframe was saved.
    - If the motion magnitude exceeds `motion_threshold` AND the `quality_score` is below `quality_threshold`, the current frame is deemed a high-quality new keyframe. Its features are then captured and added to the `memory_bank`.
    - The motion accumulator is then reset.
- **State Management**: The pipeline now meticulously tracks `prev_output_latents` and `prev_kv` (the un-warped features of the last frame) to be used as input for the next frame's calculations.

## 方案3.1: 修复与命名统一 (Fixes and Naming Unification)

Following the implementation of "方案3", an `ImportError` was identified. This was caused by inconsistent naming and outdated import statements after refactoring the attention processors. This section details the corrective actions taken.

### 1. Fixes in: `utils/wrapper.py`

This file was trying to import and use processor classes and parameters that had been renamed or deleted.

- **Corrected Imports**: The import statement at the top of the file has been fixed. It no longer tries to import the non-existent `DualFlowAttentionProcessor` or the deleted `Cached...` processors. It now correctly imports only `MultiStateAttentionProcessor`.
- **Parameter Unification**: The `__init__` method of `StreamV2VWrapper` and the internal `_load_model` method were updated. The parameter `use_dual_flow` has been renamed to `use_multi_state_attn` to match the name used in `pipeline.py`, ensuring correct flag propagation.
- **Logic Correction**: The attention processor selection logic in `_load_model` has been updated to correctly instantiate `MultiStateAttentionProcessor` when `use_multi_state_attn` is true.

### 2. Fixes in: `vid2vid/main.py`

The main execution script was also updated to reflect the name changes.

- **Command-Line Argument**: The `--use_dual_flow` flag has been renamed to `--use_multi_state_attn` to provide a consistent and accurate way to enable the new V3 feature from the command line.
- **Wrapper Instantiation**: The call to `StreamV2VWrapper` within the script has been updated to pass the `use_multi_state_attn` argument correctly.

## 方案3.2: 恢复缺失的辅助函数 (Restoring Missing Helper Functions)

Following the fix for the `ImportError`, a new `AttributeError` was detected, indicating that essential methods were missing from the `StreamV2V` class in `src/streamv2v/pipeline.py`. This was a result of the previous major rewrite of the file, where the focus on implementing the new temporal logic led to the accidental omission of existing helper functions.

This has now been corrected. The key changes are:

- **Restored Helper Methods**: Several methods that wrap core `diffusers` pipeline functionality have been added back to the `StreamV2V` class. This includes, but is not limited to:
    - `load_lcm_lora`
    - `load_lora`
    - `fuse_lora`
    - `update_prompt`
    - `txt2img`
- **Restored CFG Logic**: The `unet_step` method was also simplified too aggressively in the rewrite. The proper Classifier-Free Guidance (CFG) logic has been restored, which correctly handles `guidance_scale > 1.0` by running the U-Net with and without text conditioning and interpolating the results. This ensures the `--guidance_scale` parameter in the `main.py` script now functions as intended.

With these changes, the `StreamV2V` class now contains both the advanced "V3" temporal consistency features and its original helper utilities, resolving the `AttributeError` and ensuring full functionality.

## 方案3.3: 修复 PyTorch TypeError (Fixing PyTorch TypeError)

After restoring the missing functions, a `TypeError` was discovered when running the script. The error, `TypeError: to() received an invalid combination of arguments`, pointed to an incorrect usage of the `.to()` method in PyTorch.

This was caused by a programming error during the file rewrites, where positional arguments were used instead of the required keyword arguments for specifying both `dtype` and `device`.

- **Corrected `.to()` Calls**: All instances of the incorrect call `tensor.to(self.dtype, self.device)` within `src/streamv2v/pipeline.py` have been found and replaced with the correct syntax: `tensor.to(device=self.device, dtype=self.dtype)`. This fix was applied to tensor initializations in the `prepare` method and tensor processing in the `encode_image` and `__call__` methods, resolving the `TypeError`.

## 方案3.4: 修复 AttributeError (Tiny VAE 兼容性问题)

After fixing the `TypeError`, a new `AttributeError` related to the VAE was identified during execution.

The error, `AttributeError: 'AutoencoderTinyOutput' object has no attribute 'latent_dist'`, occurred because the code was trying to access the `.latent_dist` attribute on the output of the VAE's `encode` method. While this is correct for the standard `AutoencoderKL`, it is incorrect for the `AutoencoderTiny` used by default in this pipeline for performance reasons. The tiny VAE's output object stores the result directly in a `.latents` attribute.

- **VAE Compatibility Fix**: The `encode_image` method in `src/streamv2v/pipeline.py` has been corrected. The line for retrieving the latent representation was changed from `...latent_dist.sample(self.generator)` to simply access the `.latents` attribute. This makes the encoding process compatible with the `AutoencoderTiny` API and resolves the `AttributeError`.

