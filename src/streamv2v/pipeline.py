# pipeline.py with restored helper methods AND CORRECT BATCHING FOR CFG

import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal
from collections import deque

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from diffusers import LCMScheduler, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents

from .image_utils import postprocess_image
from .models.attention_processor import MultiStateAttentionProcessor
from . import flow_utils

class StreamV2V:
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        strength: float,
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        use_multi_state_attn: bool = True,
        memory_bank_size: int = 5,
        motion_threshold: float = 2.0,
        quality_threshold: float = 0.2,
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None
        self.frame_counter = 0

        self.height = height
        self.width = width
        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)

        self.frame_bff_size = frame_buffer_size
        self.strength = strength
        self.cfg_type = cfg_type
        self.use_denoising_batch = use_denoising_batch

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)
        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.use_multi_state_attn = use_multi_state_attn
        if self.use_multi_state_attn:
            print("Multi-State Attention is enabled.")
            self.flow_model = flow_utils.get_flow_model(self.device)
            self.memory_bank = deque(maxlen=memory_bank_size)
            self.static_anchor_kv = None
            self.motion_threshold = motion_threshold
            self.quality_threshold = quality_threshold
            self.prev_output_latents = None
            self.prev_kv = None
            self.flow_accum_since_keyframe = None

    # Restored Methods
    def load_lcm_lora(self, pretrained_model_name_or_path_or_dict, adapter_name='lcm', **kwargs) -> None:
        self.pipe.load_lora_weights(pretrained_model_name_or_path_or_dict, adapter_name, **kwargs)

    def load_lora(self, pretrained_lora_model_name_or_path_or_dict, adapter_name=None, **kwargs) -> None:
        self.pipe.load_lora_weights(pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs)

    def fuse_lora(self, fuse_unet=True, fuse_text_encoder=True, lora_scale=1.0, safe_fusing=False) -> None:
        self.pipe.fuse_lora(fuse_unet=fuse_unet, fuse_text_encoder=fuse_text_encoder, lora_scale=lora_scale, safe_fusing=safe_fusing)
    
    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        # Re-encode prompt without batching; batching done in unet_step
        do_classifier_free_guidance = self.guidance_scale > 1.0
        encoder_output = self.pipe.encode_prompt(prompt, self.device, 1, do_classifier_free_guidance)
        self.prompt_embeds_cond = encoder_output[0] # (1, 77, 768)
        if do_classifier_free_guidance:
            self.prompt_embeds_uncond = encoder_output[1] # (1, 77, 768)
        else:
            self.prompt_embeds_uncond = None

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 1.2,
        seed: int = 2,
        strength: float = 0.6,
    ) -> None:
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.guidance_scale = 1.0 if self.cfg_type == "none" else guidance_scale

        do_classifier_free_guidance = self.guidance_scale > 1.0
        encoder_output = self.pipe.encode_prompt(prompt, self.device, 1, do_classifier_free_guidance, negative_prompt)
        
        self.prompt_embeds_cond = encoder_output[0]
        if do_classifier_free_guidance:
            self.prompt_embeds_uncond = encoder_output[1]
        else:
            self.prompt_embeds_uncond = None

        # 1. Set timesteps using the correct number of steps
        self.scheduler.set_timesteps(num_inference_steps, self.device)

        # 2. Calculate offset and starting timestep based on strength (standard LCM logic)
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep_idx = int(num_inference_steps * strength) + offset
        init_timestep_idx = min(init_timestep_idx, num_inference_steps)
        
        # The latent needs to be created with noise for the *first* timestep in the schedule
        self.latent_timestep = self.scheduler.timesteps[init_timestep_idx]

        # 3. Get the actual timesteps to run on, slicing from the calculated start
        self.sub_timesteps = self.scheduler.timesteps[init_timestep_idx:].tolist()
        self.sub_timesteps_tensor = self.scheduler.timesteps[init_timestep_idx:]

        # 4. Recalculate batch size and prepare scheduler constants for the *actual* number of steps
        self.denoising_steps_num = len(self.sub_timesteps)
        self.batch_size = self.denoising_steps_num * self.frame_bff_size if self.use_denoising_batch else self.frame_bff_size
        
        if self.batch_size > 0:
            self.init_noise = torch.randn(
                (self.batch_size, 4, self.latent_height, self.latent_width),
                generator=self.generator, device=self.device, dtype=self.dtype
            )

            c_skip_list, c_out_list, alpha_list, beta_list = [], [], [], []
            for t in self.sub_timesteps:
                c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(t)
                alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[t].sqrt()
                beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[t]).sqrt()
                c_skip_list.append(torch.tensor(c_skip, device=self.device, dtype=self.dtype))
                c_out_list.append(torch.tensor(c_out, device=self.device, dtype=self.dtype))
                alpha_list.append(alpha_prod_t_sqrt)
                beta_list.append(beta_prod_t_sqrt)

            self.c_skip = torch.stack(c_skip_list).view(-1, 1, 1, 1).to(device=self.device, dtype=self.dtype)
            self.c_out = torch.stack(c_out_list).view(-1, 1, 1, 1).to(device=self.device, dtype=self.dtype)
            self.alpha_prod_t_sqrt = torch.stack(alpha_list).view(-1, 1, 1, 1).to(device=self.device, dtype=self.dtype)
            self.beta_prod_t_sqrt = torch.stack(beta_list).view(-1, 1, 1, 1).to(device=self.device, dtype=self.dtype)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)
        return self.scheduler.add_noise(x, noise, self.latent_timestep)

    def unet_step(self, x_t: torch.Tensor, t_input: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Performs a single UNet denoising step.
        x_t: Latent input to the UNet (batch_size_actual, C, H, W)
        t_input: Timesteps for this batch (batch_size_actual,)
        """
        current_latent_batch_size = x_t.shape[0]

        if self.guidance_scale > 1.0:
            latent_model_input = torch.cat([x_t] * 2)
            # Repeat uncond and cond prompt embeds to match current_latent_batch_size * 2
            prompt_embeds_for_unet = torch.cat([
                self.prompt_embeds_uncond.repeat(current_latent_batch_size, 1, 1),
                self.prompt_embeds_cond.repeat(current_latent_batch_size, 1, 1)
            ])
            # Repeat timesteps to match new batch size
            t_for_unet = t_input.repeat(2)
        else:
            latent_model_input = x_t
            # Repeat cond prompt embeds to match current_latent_batch_size
            prompt_embeds_for_unet = self.prompt_embeds_cond.repeat(current_latent_batch_size, 1, 1)
            t_for_unet = t_input
        
        # Slicing the alpha/beta/c_skip/c_out to match the actual batch size of x_t
        # This is needed because self.alpha_prod_t_sqrt etc. might be of size self.batch_size (e.g. 4)
        # but x_t here might be a smaller batch (e.g., 1 or 2).
        alpha_prod_t_sqrt_slice = self.alpha_prod_t_sqrt[:x_t.shape[0]]
        beta_prod_t_sqrt_slice = self.beta_prod_t_sqrt[:x_t.shape[0]]
        c_out_slice = self.c_out[:x_t.shape[0]]
        c_skip_slice = self.c_skip[:x_t.shape[0]]

        model_pred = self.unet(
            latent_model_input,
            t_for_unet, # Use batch-matched timesteps
            encoder_hidden_states=prompt_embeds_for_unet, # Use batch-matched prompt embeds
            cross_attention_kwargs=kwargs, 
            return_dict=False
        )[0]
        
        if self.guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
            model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        F_theta = (x_t - beta_prod_t_sqrt_slice * model_pred) / alpha_prod_t_sqrt_slice
        return c_out_slice * F_theta + c_skip_slice * x_t

    def _get_kv_from_unet(self):
        kv_map = {}
        # Sort by name to ensure order is consistent, which is important for reconstructing the KV storage
        sorted_processors = sorted(self.unet.attn_processors.items())
        for name, processor in sorted_processors:
            if isinstance(processor, MultiStateAttentionProcessor):
                if processor.last_key is not None and processor.last_value is not None:
                    # Store on CPU to save VRAM
                    kv_map[name] = (processor.last_key.cpu(), processor.last_value.cpu())
        return kv_map

    def encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_latent = self.vae.encode(img_tensor.to(device=self.device, dtype=self.dtype)).latents
        img_latent *= self.vae.config.scaling_factor
        return self.add_noise(img_latent)

    def decode_image(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

    def predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        cross_attention_kwargs = {}
        flow_src = None # Initialize flow_src
        if self.use_multi_state_attn and self.frame_counter > 0:
            # --- Prepare per-layer KVs for the Attention Processor ---
            prev_kvs_by_layer = {}
            bank_kvs_by_layer = {}

            # 1. Calculate source flow
            flow_src = flow_utils.calculate_flow(self.prev_output_latents, x_t_latent, self.flow_model, self.vae, self.vae.config.scaling_factor)

            # 2. Warp previous KVs
            for name, (k_prev, v_prev) in self.prev_kv.items():
                k_prev = k_prev.to(self.device)
                v_prev = v_prev.to(self.device)

                # Mathematically derive h, w from seq_len and aspect ratio
                b, seq_len, f = k_prev.shape
                aspect_ratio = self.latent_width / self.latent_height
                h = int((seq_len / aspect_ratio)**0.5)
                w = round(seq_len / h)

                # Reshape K, warp, and reshape back
                k_prev_4d = k_prev.transpose(1, 2).view(b, f, h, w)
                k_prev_warped_4d = flow_utils.warp(k_prev_4d, flow_src)
                k_prev_warped = k_prev_warped_4d.view(b, f, seq_len).transpose(1, 2)

                # Reshape V, warp, and reshape back
                v_prev_4d = v_prev.transpose(1, 2).view(b, f, h, w)
                v_prev_warped_4d = flow_utils.warp(v_prev_4d, flow_src)
                v_prev_warped = v_prev_warped_4d.view(b, f, seq_len).transpose(1, 2)

                prev_kvs_by_layer[name] = (k_prev_warped, v_prev_warped)

            # 3. Assemble memory bank KVs
            static_kvs = {name: (k.to(self.device), v.to(self.device)) for name, (k, v) in self.static_anchor_kv.items()}
            
            dynamic_kvs_list = []
            for bank_item in self.memory_bank:
                dynamic_kvs_list.append({name: (k.to(self.device), v.to(self.device)) for name, (k, v) in bank_item.items()})

            for name in static_kvs.keys():
                static_k, static_v = static_kvs[name]
                dynamic_ks = [d[name][0] for d in dynamic_kvs_list]
                dynamic_vs = [d[name][1] for d in dynamic_kvs_list]
                
                bank_k = torch.cat([static_k] + dynamic_ks, dim=1)
                bank_v = torch.cat([static_v] + dynamic_vs, dim=1)
                bank_kvs_by_layer[name] = (bank_k, bank_v)

            # 4. Prepare discrepancy mask and final kwargs
            flow_gen_approx = flow_src
            discrepancy_mask = torch.norm(flow_src - flow_gen_approx, p=2, dim=1, keepdim=True)
            discrepancy_mask = (discrepancy_mask - discrepancy_mask.min()) / (discrepancy_mask.max() - discrepancy_mask.min() + 1e-6)
            
            cross_attention_kwargs = {
                "discrepancy_mask": discrepancy_mask,
                "bank_kvs_by_layer": bank_kvs_by_layer,
                "prev_kvs_by_layer": prev_kvs_by_layer,
            }

        # Handle denoising batch logic here
        if self.use_denoising_batch and not self.use_multi_state_attn:
            x_t_latent_input = x_t_latent.repeat(self.batch_size, 1, 1, 1)
            t_input_for_unet_step = self.sub_timesteps_tensor
        else:
            x_t_latent_input = x_t_latent
            t_input_for_unet_step = self.sub_timesteps_tensor[:x_t_latent.shape[0]]
        
        x_0_pred_out = self.unet_step(x_t_latent_input, t_input_for_unet_step, **cross_attention_kwargs)
            
        if self.use_multi_state_attn:
            current_kv_map = self._get_kv_from_unet() # Returns a map {name: (k, v)} on CPU

            if not current_kv_map:
                print("Warning: Could not extract any KVs from UNet. Multi-state attention will be inactive.")
            else:
                if self.frame_counter == 0:
                    self.static_anchor_kv = current_kv_map
                else:
                    if flow_src is not None:
                        true_flow_gen = flow_utils.calculate_flow(self.prev_output_latents, x_0_pred_out, self.flow_model, self.vae, self.vae.config.scaling_factor)
                        final_discrepancy = torch.norm(flow_src - true_flow_gen, p=2).mean()
                        
                        if self.flow_accum_since_keyframe is None:
                            self.flow_accum_since_keyframe = flow_src
                        else:
                            self.flow_accum_since_keyframe = flow_utils.compose_flow(flow_src, self.flow_accum_since_keyframe)
                        
                        motion_magnitude = torch.norm(self.flow_accum_since_keyframe, p=2).mean()

                        if motion_magnitude > self.motion_threshold and final_discrepancy < self.quality_threshold:
                            print(f"Frame {self.frame_counter}: Adding new dynamic keyframe.")
                            self.memory_bank.append(current_kv_map)
                            self.flow_accum_since_keyframe = None

                self.prev_output_latents = x_0_pred_out.clone()
                self.prev_kv = current_kv_map
            
        self.frame_counter += 1
        return x_0_pred_out

    @torch.no_grad()
    def __call__(self, x: Union[torch.Tensor, PIL.Image.Image, np.ndarray]) -> torch.Tensor:
        if not hasattr(self, "denoising_steps_num") or self.denoising_steps_num == 0:
            return x

        x = self.image_processor.preprocess(x, self.height, self.width).to(device=self.device, dtype=self.dtype)
        x_t_latent = self.encode_image(x)
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        if self.use_denoising_batch and x_0_pred_out.shape[0] > 1:
            x_0_pred_out = x_0_pred_out.mean(dim=0, keepdim=True)
        return self.decode_image(x_0_pred_out).detach().clone()

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = torch.randn((batch_size, 4, self.latent_height, self.latent_width), device=self.device, dtype=self.dtype, generator=self.generator)
        x_0_pred_out = self.predict_x0_batch(x_t_latent)
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        return x_output