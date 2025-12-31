"""Image and video preprocessing utilities for ComfyUI integration."""

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple


def comfyui_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI IMAGE tensor to PIL Image.

    ComfyUI images are in format [B, H, W, 3] with values 0-1 (float32).

    Args:
        image_tensor: ComfyUI IMAGE tensor [B, H, W, 3]

    Returns:
        PIL Image (RGB)
    """
    # Take first image from batch if batch size > 1
    if image_tensor.dim() == 4:
        img = image_tensor[0]
    else:
        img = image_tensor

    # Convert to numpy and scale to 0-255
    img_np = img.cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    # Convert to PIL Image
    return Image.fromarray(img_np, mode="RGB")


def pil_to_tensor(image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """
    Convert PIL Image to tensor for model input.

    Args:
        image: PIL Image (RGB)
        device: Device to place tensor on

    Returns:
        Tensor in format [C, H, W] with values 0-1
    """
    # Convert to numpy array
    img_np = np.array(image).astype(np.float32) / 255.0

    # Convert HWC to CHW format
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (2, 0, 1))
    elif img_np.ndim == 2:
        # Grayscale image - add channel dimension
        img_np = np.expand_dims(img_np, 0)

    # Convert to tensor
    tensor = torch.from_numpy(img_np).to(device)

    return tensor


def pil_to_comfyui(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI IMAGE format.

    Args:
        image: PIL Image (RGB)

    Returns:
        Tensor in ComfyUI format [1, H, W, 3] with values 0-1
    """
    # Convert to numpy array
    img_np = np.array(image).astype(np.float32) / 255.0

    # Ensure RGB format
    if img_np.ndim == 2:
        # Grayscale - convert to RGB
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[-1] == 4:
        # RGBA - drop alpha channel
        img_np = img_np[:, :, :3]

    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(img_np).unsqueeze(0)

    return tensor


def video_to_comfyui(
    frames: Union[List[Image.Image], List[np.ndarray], torch.Tensor]
) -> torch.Tensor:
    """
    Convert video frames to ComfyUI IMAGE batch format.

    Args:
        frames: List of PIL Images, numpy arrays [H,W,3], or torch tensor

    Returns:
        Tensor in ComfyUI format [N, H, W, 3] with values 0-1
    """
    if isinstance(frames, torch.Tensor):
        # Already a tensor, ensure correct format
        if frames.dim() == 4 and frames.shape[-1] == 3:
            # Already in [N,H,W,3] format
            if frames.max() > 1.0:
                frames = frames.float() / 255.0
            return frames
        elif frames.dim() == 4 and frames.shape[1] == 3:
            # In [N,C,H,W] format - need to convert to [N,H,W,C]
            frames = frames.permute(0, 2, 3, 1)
            if frames.max() > 1.0:
                frames = frames.float() / 255.0
            return frames

    frame_arrays = []

    for frame in frames:
        if isinstance(frame, Image.Image):
            # PIL Image
            frame_np = np.array(frame).astype(np.float32) / 255.0
        elif isinstance(frame, np.ndarray):
            # Numpy array
            frame_np = frame.astype(np.float32)
            if frame_np.max() > 1.0:
                frame_np = frame_np / 255.0
        elif isinstance(frame, torch.Tensor):
            # Single frame tensor
            frame_np = frame.cpu().numpy().astype(np.float32)
            if frame_np.max() > 1.0:
                frame_np = frame_np / 255.0
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")

        # Ensure RGB format [H, W, 3]
        if frame_np.ndim == 2:
            # Grayscale
            frame_np = np.stack([frame_np] * 3, axis=-1)
        elif frame_np.ndim == 3:
            if frame_np.shape[0] == 3:
                # CHW format - convert to HWC
                frame_np = np.transpose(frame_np, (1, 2, 0))
            if frame_np.shape[-1] == 4:
                # RGBA - drop alpha
                frame_np = frame_np[:, :, :3]

        frame_arrays.append(frame_np)

    # Stack frames into batch [N, H, W, 3]
    batch = np.stack(frame_arrays, axis=0)

    # Convert to torch tensor
    return torch.from_numpy(batch)


def comfyui_to_video_tensor(image_batch: torch.Tensor) -> torch.Tensor:
    """
    Convert ComfyUI IMAGE batch to video tensor for model input.

    Args:
        image_batch: ComfyUI IMAGE batch [N, H, W, 3] with values 0-1

    Returns:
        Video tensor in format [N, C, H, W]
    """
    # Permute from NHWC to NCHW
    video_tensor = image_batch.permute(0, 3, 1, 2)
    return video_tensor


def resize_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    maintain_aspect: bool = True,
) -> Image.Image:
    """
    Resize image to target size.

    Args:
        image: PIL Image
        target_size: (width, height) tuple
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        # Calculate aspect-preserving size
        img_aspect = image.width / image.height
        target_aspect = target_size[0] / target_size[1]

        if img_aspect > target_aspect:
            # Image is wider - fit to width
            new_width = target_size[0]
            new_height = int(new_width / img_aspect)
        else:
            # Image is taller - fit to height
            new_height = target_size[1]
            new_width = int(new_height * img_aspect)

        # Resize and then pad/crop to exact size if needed
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with target size and paste resized image centered
        if (new_width, new_height) != target_size:
            new_image = Image.new("RGB", target_size, (0, 0, 0))
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            new_image.paste(image, (paste_x, paste_y))
            return new_image

        return image
    else:
        # Direct resize without maintaining aspect ratio
        return image.resize(target_size, Image.Resampling.LANCZOS)


def get_resolution_size(resolution: str) -> Tuple[int, int]:
    """
    Get pixel dimensions for resolution string.

    Args:
        resolution: Resolution string (e.g., "720p", "480p")

    Returns:
        (width, height) tuple
    """
    resolution_map = {
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
    }

    if resolution not in resolution_map:
        raise ValueError(
            f"Unknown resolution: {resolution}. "
            f"Available: {list(resolution_map.keys())}"
        )

    return resolution_map[resolution]


def normalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Normalize tensor with mean and std.

    Args:
        tensor: Input tensor [C, H, W] or [N, C, H, W]
        mean: Mean values for each channel
        std: Std values for each channel

    Returns:
        Normalized tensor
    """
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return (tensor - mean) / std


def denormalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Denormalize tensor with mean and std.

    Args:
        tensor: Normalized tensor [C, H, W] or [N, C, H, W]
        mean: Mean values for each channel
        std: Std values for each channel

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean
