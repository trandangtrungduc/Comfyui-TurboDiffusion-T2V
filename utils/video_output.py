"""Video output utilities for saving frames as video files."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
from PIL import Image

# Optional imports for video export
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def save_video_cv2(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 24,
    codec: str = "mp4v",
) -> None:
    """
    Save frames as video using OpenCV.

    Args:
        frames: Frame tensor [N, H, W, 3] with values 0-1
        output_path: Output video file path
        fps: Frames per second
        codec: Video codec (mp4v, avc1, etc.)

    Raises:
        ImportError: If opencv-python is not installed
    """
    if not HAS_CV2:
        raise ImportError(
            "opencv-python is required for MP4 export. "
            "Install with: pip install opencv-python"
        )

    # Convert to numpy
    if isinstance(frames, torch.Tensor):
        frames_np = frames.cpu().numpy()
    else:
        frames_np = frames

    # Scale to 0-255 and convert to uint8
    if frames_np.max() <= 1.0:
        frames_np = (frames_np * 255).astype(np.uint8)
    else:
        frames_np = frames_np.astype(np.uint8)

    # Get dimensions
    n_frames, height, width, channels = frames_np.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    output_path = str(output_path)

    # Ensure .mp4 extension
    if not output_path.endswith('.mp4'):
        output_path += '.mp4'

    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    # Write frames
    for i in range(n_frames):
        frame = frames_np[i]
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()
    print(f"Video saved to: {output_path}")


def save_video_imageio(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 24,
    quality: int = 8,
) -> None:
    """
    Save frames as video using imageio.

    Args:
        frames: Frame tensor [N, H, W, 3] with values 0-1
        output_path: Output video file path
        fps: Frames per second
        quality: Video quality (1-10, higher is better)

    Raises:
        ImportError: If imageio is not installed
    """
    if not HAS_IMAGEIO:
        raise ImportError(
            "imageio and imageio-ffmpeg are required for video export. "
            "Install with: pip install imageio imageio-ffmpeg"
        )

    # Convert to numpy
    if isinstance(frames, torch.Tensor):
        frames_np = frames.cpu().numpy()
    else:
        frames_np = frames

    # Scale to 0-255 and convert to uint8
    if frames_np.max() <= 1.0:
        frames_np = (frames_np * 255).astype(np.uint8)
    else:
        frames_np = frames_np.astype(np.uint8)

    # Ensure .mp4 extension
    output_path = str(output_path)
    if not output_path.endswith('.mp4'):
        output_path += '.mp4'

    # Save video
    imageio.mimsave(
        output_path,
        frames_np,
        fps=fps,
        quality=quality,
        macro_block_size=None,
    )
    print(f"Video saved to: {output_path}")


def save_gif(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 24,
    loop: int = 0,
    optimize: bool = True,
) -> None:
    """
    Save frames as GIF.

    Args:
        frames: Frame tensor [N, H, W, 3] with values 0-1
        output_path: Output GIF file path
        fps: Frames per second
        loop: Number of loops (0 = infinite)
        optimize: Whether to optimize GIF size
    """
    # Convert to numpy
    if isinstance(frames, torch.Tensor):
        frames_np = frames.cpu().numpy()
    else:
        frames_np = frames

    # Scale to 0-255 and convert to uint8
    if frames_np.max() <= 1.0:
        frames_np = (frames_np * 255).astype(np.uint8)
    else:
        frames_np = frames_np.astype(np.uint8)

    # Convert to list of PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames_np]

    # Ensure .gif extension
    output_path = str(output_path)
    if not output_path.endswith('.gif'):
        output_path += '.gif'

    # Calculate duration in milliseconds
    duration = int(1000 / fps)

    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop,
        optimize=optimize,
    )
    print(f"GIF saved to: {output_path}")


def save_video(
    frames: torch.Tensor,
    output_path: str,
    fps: int = 24,
    format: str = "mp4",
    **kwargs,
) -> None:
    """
    Save frames as video in specified format.

    Args:
        frames: Frame tensor [N, H, W, 3] with values 0-1
        output_path: Output file path (extension will be added if missing)
        fps: Frames per second
        format: Output format (mp4, gif, webm)
        **kwargs: Additional format-specific arguments

    Raises:
        ValueError: If format is not supported
        ImportError: If required package is not installed
    """
    output_path = Path(output_path)

    # Remove extension if present
    output_path = output_path.with_suffix('')

    format = format.lower()

    if format == "mp4":
        # Try imageio first, fallback to cv2
        if HAS_IMAGEIO:
            save_video_imageio(
                frames,
                str(output_path) + ".mp4",
                fps=fps,
                quality=kwargs.get("quality", 8),
            )
        elif HAS_CV2:
            save_video_cv2(
                frames,
                str(output_path) + ".mp4",
                fps=fps,
                codec=kwargs.get("codec", "mp4v"),
            )
        else:
            raise ImportError(
                "Either opencv-python or imageio is required for MP4 export. "
                "Install with: pip install opencv-python OR pip install imageio imageio-ffmpeg"
            )

    elif format == "gif":
        save_gif(
            frames,
            str(output_path) + ".gif",
            fps=fps,
            loop=kwargs.get("loop", 0),
            optimize=kwargs.get("optimize", True),
        )

    elif format == "webm":
        if not HAS_IMAGEIO:
            raise ImportError(
                "imageio is required for WebM export. "
                "Install with: pip install imageio imageio-ffmpeg"
            )
        # Use imageio for webm
        frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)

        imageio.mimsave(
            str(output_path) + ".webm",
            frames_np,
            fps=fps,
            codec="libvpx-vp9",
            quality=kwargs.get("quality", 8),
        )
        print(f"WebM video saved to: {output_path}.webm")

    else:
        raise ValueError(
            f"Unsupported format: {format}. "
            f"Available formats: mp4, gif, webm"
        )


def save_frames_as_images(
    frames: torch.Tensor,
    output_dir: str,
    prefix: str = "frame",
    format: str = "png",
) -> List[str]:
    """
    Save individual frames as image files.

    Args:
        frames: Frame tensor [N, H, W, 3] with values 0-1
        output_dir: Output directory
        prefix: Filename prefix
        format: Image format (png, jpg, etc.)

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Convert to numpy
    if isinstance(frames, torch.Tensor):
        frames_np = frames.cpu().numpy()
    else:
        frames_np = frames

    # Scale to 0-255
    if frames_np.max() <= 1.0:
        frames_np = (frames_np * 255).astype(np.uint8)
    else:
        frames_np = frames_np.astype(np.uint8)

    saved_paths = []

    for i, frame in enumerate(frames_np):
        # Create filename with zero-padded frame number
        filename = f"{prefix}_{i:06d}.{format}"
        filepath = output_dir / filename

        # Convert to PIL and save
        img = Image.fromarray(frame)
        img.save(filepath)
        saved_paths.append(str(filepath))

    print(f"Saved {len(saved_paths)} frames to: {output_dir}")
    return saved_paths
