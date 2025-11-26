import gc
import os
from pathlib import Path
import traceback
from typing import List, Literal, Optional, Union, Dict

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
from PIL import Image
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.streamv2v import StreamV2V
from src.streamv2v.image_utils import postprocess_image
from src.streamv2v.models.attention_processor import MultiStateAttentionProcessor


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class StreamV2VWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        strength: float,
        lora_dict: Optional[Dict[str, float]] = None,
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        mode: Literal["img2img", "txt2img"] = "img2img",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "none",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "none",
        use_multi_state_attn: bool = False,
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
        memory_bank_size: int = 5,
        motion_threshold: float = 2.0,
        quality_threshold: float = 0.2,
    ):
        self.sd_turbo = "turbo" in model_id_or_path
        self.sd_xl = "xl" in model_id_or_path

        if mode == "txt2img" and cfg_type != "none":
            raise ValueError(f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}")
        self.mode = mode

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        
        # This will be recalculated in the pipeline, so a rough estimate is fine for the warmup loop
        self.batch_size = (int(strength * 50) * frame_buffer_size if use_denoising_batch else frame_buffer_size)

        self.use_denoising_batch = use_denoising_batch
        self.use_multi_state_attn = use_multi_state_attn
        self.use_safety_checker = use_safety_checker
        self.acceleration = acceleration # Store acceleration setting

        self.stream: StreamV2V = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            strength=strength,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
            use_multi_state_attn=self.use_multi_state_attn,
            memory_bank_size=memory_bank_size,
            motion_threshold=motion_threshold,
            quality_threshold=quality_threshold,
        )

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(self.stream.unet, device_ids=device_ids)

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(similar_image_filter_threshold, similar_image_filter_max_skip_frame)

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        strength: float = 0.5,
    ) -> None:
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if prompt is not None:
            self.stream.update_prompt(prompt)
        image_tensor = self.stream.txt2img(self.frame_buffer_size)
        return self.postprocess_image(image_tensor, output_type=self.output_type)

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if prompt is not None:
            self.stream.update_prompt(prompt)
        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)
        image_tensor = self.stream(image)
        return self.postprocess_image(image_tensor, output_type=self.output_type)

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))
        return self.stream.image_processor.preprocess(image, self.height, self.width).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)
        else:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)[0]

    def _load_model(
        self,
        model_id_or_path: str,
        strength: float,
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "tensorrt", "sfast"] = "none",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
        use_multi_state_attn: bool = False,
        **kwargs,
    ) -> StreamV2V:
        pipeline_cls = StableDiffusionXLPipeline if self.sd_xl else StableDiffusionPipeline
        try:
            pipe = pipeline_cls.from_pretrained(model_id_or_path).to(device=self.device, dtype=self.dtype)
        except ValueError:
            pipe = pipeline_cls.from_single_file(model_id_or_path).to(device=self.device, dtype=self.dtype)
        except Exception as e:
            traceback.print_exc()
            sys.exit(f"Failed to load model: {e}")

        if self.sd_xl:
            pipe.unet.config.addition_embed_type = None
            
        stream = StreamV2V(
            pipe=pipe,
            strength=strength,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
            use_multi_state_attn=use_multi_state_attn,
            **kwargs,
        )
        if not self.sd_turbo and use_lcm_lora:
            stream.load_lcm_lora(lcm_lora_id or "latent-consistency/lcm-lora-sdv1-5", "lcm")
        if lora_dict:
            for name, scale in lora_dict.items():
                stream.load_lora(name)
        if use_tiny_vae:
            stream.vae = AutoencoderTiny.from_pretrained(vae_id or "madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

        try:
            # --- Acceleration / Attention Processor Setup ---
            if self.use_multi_state_attn:
                print("Enabling Multi-State Attention Processor...")
                # Force diffusers to populate the attn_processors dictionary by setting a dummy processor first.
                # This gives us the keys to iterate over.
                stream.pipe.unet.set_attn_processor(AttnProcessor())
                
                # Now that the keys are populated, we can override them with our custom processor.
                attn_processors = {}
                for name in stream.pipe.unet.attn_processors.keys():
                    attn_processors[name] = MultiStateAttentionProcessor(name=name)
                stream.pipe.unet.set_attn_processor(attn_processors)

                if acceleration != "none":
                    print(f"Warning: Multi-State Attention is active. Disabling '{acceleration}' acceleration.")
                    acceleration = "none"
            elif acceleration == "tensorrt":
                raise NotImplementedError("TensorRT acceleration is not compatible with Multi-State Attention.")
            elif acceleration == "sfast":
                raise NotImplementedError("StableFast acceleration is not compatible with Multi-State Attention.")
            # If acceleration is "xformers" but use_multi_state_attn is false, nothing will happen here,
            # as the default acceleration is now "none".
            # The user explicitly asked to disable xformers, so we don't need a specific elif for it.
            # If they *want* xformers without multi_state_attn, they'd have to explicitly enable it elsewhere.
            # --- End Acceleration / Attention Processor Setup ---

        except Exception:
            traceback.print_exc()
            print("Acceleration/Attention Processor setup has failed. Falling back to normal mode.")
            # Ensure MultiStateAttention is disabled if setup fails
            stream.use_multi_state_attn = False
            # Revert UNet processors to default if custom setup failed
            stream.pipe.unet.set_attn_processor({}) # Set to empty dict to revert to default AttnProcessor

        if seed < 0:
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "", "", num_inference_steps=50,
            guidance_scale=1.2 if stream.cfg_type in ["full", "self", "initialize"] else 1.0,
            seed=seed,
            strength=strength
        )

        return stream