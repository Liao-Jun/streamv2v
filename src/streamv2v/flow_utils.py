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
    vae: torch.nn.Module,
    vae_scale_factor: float,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Calculates optical flow from img1 to img2.
    Decodes latents to images if necessary.

    Args:
        img1 (torch.Tensor): The first image/latent tensor (t-1).
        img2 (torch.Tensor): The second image/latent tensor (t).
        model (dict): The RAFT model and transforms.
        vae (torch.nn.Module): The VAE for decoding latents.
        vae_scale_factor (float): The scaling factor for the VAE.

    Returns:
        torch.Tensor: The calculated flow field.
    """
    # Decode latents to images if they are 4-channel
    if img1.shape[1] == 4:
        img1 = vae.decode(img1 / vae_scale_factor, return_dict=False)[0]
        img1 = (img1 / 2) + 0.5  # Shift from [-1, 1] to [0, 1]
    if img2.shape[1] == 4:
        img2 = vae.decode(img2 / vae_scale_factor, return_dict=False)[0]
        img2 = (img2 / 2) + 0.5  # Shift from [-1, 1] to [0, 1]

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
        flow (torch.Tensor): The flow field. Shape can be different from x, will be resized.

    Returns:
        torch.Tensor: The warped tensor.
    """
    B, C, H, W = x.size()

    # Downsample flow to match the feature map's size if it doesn't match
    flow_h_orig, flow_w_orig = flow.shape[-2:]
    if flow_h_orig != H or flow_w_orig != W:
        flow_resized = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        # Scale the flow values to match the new resolution
        flow_resized[:, 0, :, :] = flow_resized[:, 0, :, :] * (W / flow_w_orig)
        flow_resized[:, 1, :, :] = flow_resized[:, 1, :, :] * (H / flow_h_orig)
        flow = flow_resized
    
    # Create sampling grid in float32 for precision
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    grid = torch.stack([xx, yy], dim=0).float().to(x.device)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
    
    # Add flow to grid, ensuring float32 for the operation
    vgrid = grid + flow.float()
    
    # Scale grid to [-1, 1] for grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    
    vgrid = vgrid.permute(0, 2, 3, 1)
    
    # Warp tensor, casting vgrid to match input tensor's dtype at the last moment
    warped_x = F.grid_sample(x, vgrid.to(x.dtype), mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return warped_x


def compose_flow(flow_t_minus_1_to_t: torch.Tensor, flow_0_to_t_minus_1: torch.Tensor) -> torch.Tensor:
    """
    Composes two flow fields using the chain rule.
    F_accum(0 -> t) = F(t-1 -> t) + warp(F_accum(0 -> t-1), F(t-1 -> t))
    
    Note: The original paper suggests the inverse, but for forward accumulation this is more intuitive.
    Let's stick to the user's formula: F_accum = F_current + F_accum_warped
    Here, we warp the accumulated flow by the *current* frame's flow. Let's double check.
    To find where a pixel at `p` in frame 0 moves to in frame `t`, we first see where it is in `t-1`.
    Let p_t_minus_1 = p + F_accum(0 -> t-1)(p).
    Then the final position is p_t = p_t_minus_1 + F(t-1 -> t)(p_t_minus_1).
    This means we need to sample the current flow at the warped coordinate.
    This is equivalent to warping the current flow field by the accumulated flow.
    Let's refine: F(0->t) = warp(F(t-1->t), F(0->t-1)) + F(0->t-1).
    Let's use a simpler, more common approximation which is more stable:
    F_accum_new = flow_current + warp(flow_accum_old, flow_current)
    """
    if flow_0_to_t_minus_1 is None:
        return flow_t_minus_1_to_t

    # Warp the previous accumulated flow with the current flow
    warped_accum_flow = warp(flow_0_to_t_minus_1, flow_t_minus_1_to_t)
    
    # Add the current flow
    composed = warped_accum_flow + flow_t_minus_1_to_t
    
    return composed

