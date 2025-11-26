import os
import sys
import time
import argparse
from typing import Literal

import numpy as np
import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.wrapper import StreamV2VWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Perform video-to-video translation with StreamV2V.")
    
    # Core arguments
    parser.add_argument("--input", type=str, required=True, help="The input video file path.")
    parser.add_argument("--prompt", type=str, required=True, help="The editing prompt to perform video translation.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(CURRENT_DIR, "outputs"), help="The directory for the output video.")
    parser.add_argument("--model_id", type=str, default="Jiali/stable-diffusion-1.5", help="The base image diffusion model.")
    
    # Editing strength and guidance arguments
    parser.add_argument("--strength", type=float, default=0.5, help="Edit strength (0.0 to 1.0). Higher values mean more significant edits. This corresponds to the amount of noise added.")
    parser.add_argument("--guidance_scale", type=float, default=1.2, help="Classifier-Free Guidance scale. Higher values align the output more closely with the prompt. Must be > 1.0 to be effective.")
    parser.add_argument("--cfg_type", type=str, default="self", choices=["none", "full", "self", "initialize"], help="Type of CFG to use. 'self' or 'full' are recommended with guidance_scale > 1.")

    # Performance and feature arguments
    parser.add_argument("--diffusion_steps", type=int, default=4, help="Number of diffusion steps. Higher steps usually lead to higher quality but slower speed.")
    parser.add_argument("--scale", type=float, default=1.0, help="The scale of the resolution.")
    parser.add_argument("--acceleration", type=str, default="none", choices=["none", "tensorrt", "sfast"], help="Type of acceleration to use.")
    parser.add_argument("--no_denoising_batch", action="store_false", dest="use_denoising_batch", help="Disable the denoising batch feature (recommended when using multi-state attention).")
    parser.add_argument("--seed", type=int, default=42, help="The seed for random number generation. -1 for random.")

    # Temporal consistency arguments
    parser.add_argument("--use_multi_state_attn", action="store_true", help="Enable the Multi-State Attention mechanism for improved temporal consistency.")
    parser.add_argument("--memory_bank_size", type=int, default=2, help="The number of keyframes to store in the memory bank for temporal consistency.")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.use_multi_state_attn and args.use_denoising_batch:
        print("Warning: use_denoising_batch is incompatible with use_multi_state_attn. Disabling use_denoising_batch.")
        args.use_denoising_batch = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    video_info = read_video(args.input)
    video = video_info[0] / 255
    fps = video_info[2]["video_fps"]
    height = int(video.shape[1] * args.scale)
    width = int(video.shape[2] * args.scale)

    print(f"Using Multi-State-Attention: {args.use_multi_state_attn}")
    print(f"Using strength {args.strength}, CFG scale {args.guidance_scale} with type '{args.cfg_type}'.")
    print(f"Running for {args.diffusion_steps} steps.")

    stream = StreamV2VWrapper(
        model_id_or_path=args.model_id,
        strength=args.strength,
        mode="img2img",
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=args.acceleration,
        output_type="pt",
        use_denoising_batch=args.use_denoising_batch,
        cfg_type=args.cfg_type,
        use_multi_state_attn=args.use_multi_state_attn,
        memory_bank_size=args.memory_bank_size,
        seed=args.seed,
    )
    stream.prepare(
        prompt=args.prompt,
        num_inference_steps=args.diffusion_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
    )

    video_result = torch.zeros(video.shape[0], height, width, 3)

    # Warmup
    for _ in range(stream.batch_size):
        stream(image=video[0].permute(2, 0, 1))

    print("Starting video processing...")
    inference_time = []
    for i in tqdm(range(video.shape[0])):
        iteration_start_time = time.time()
        output_image = stream(video[i].permute(2, 0, 1))
        video_result[i] = output_image.permute(1, 2, 0)
        iteration_end_time = time.time()
        inference_time.append(iteration_end_time - iteration_start_time)
    
    avg_fps = 1.0 / (sum(inference_time) / len(inference_time)) if inference_time else 0
    print(f"Processing finished. Average FPS: {avg_fps:.2f}")

    video_result = (video_result * 255).byte()
    prompt_txt = args.prompt.replace(' ', '-')[:50]
    input_vid = os.path.basename(args.input)
    output_filename = f"{input_vid.rsplit('.', 1)[0]}_{prompt_txt}.mp4"
    output_path = os.path.join(args.output_dir, output_filename)
    
    import imageio
    imageio.mimwrite(output_path, video_result.numpy(), fps=float(fps))
    # write_video(output_path, video_result, fps=float(fps))
    print(f"Output video saved to: {output_path}")


if __name__ == "__main__":
    main()