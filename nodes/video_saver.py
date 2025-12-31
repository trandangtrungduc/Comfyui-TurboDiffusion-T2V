"""Video saver node for ComfyUI."""

from typing import Dict
from pathlib import Path
import torch
from datetime import datetime

from ..utils.video_output import save_video


class TurboDiffusionSaveVideo:
    """
    ComfyUI node for saving frame batches as video files.

    This is an output node that saves the generated frames to disk
    in various video formats (MP4, GIF, WebM).
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node."""
        return {
            "required": {
                "frames": (
                    "IMAGE",
                    {
                        "tooltip": "Frame batch from TurboDiffusion I2V Generator",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "turbodiffusion",
                        "tooltip": "Prefix for output filename. Timestamp will be appended.",
                    },
                ),
                "fps": (
                    "INT",
                    {
                        "default": 24,
                        "min": 1,
                        "max": 60,
                        "step": 1,
                        "tooltip": "Frames per second for video playback",
                    },
                ),
                "format": (
                    ["mp4", "gif", "webm"],
                    {
                        "default": "mp4",
                        "tooltip": "Output video format. MP4 recommended for quality, GIF for sharing.",
                    },
                ),
            },
            "optional": {
                "quality": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 10,
                        "tooltip": "Video quality (1-10, higher is better). Only for MP4/WebM.",
                    },
                ),
                "optimize_gif": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Optimize GIF file size. Only for GIF format.",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "video/turbodiffusion"
    DESCRIPTION = "Save frame batch as video file (MP4, GIF, or WebM)"

    def __init__(self):
        """Initialize the video saver node."""
        self.output_dir = self._get_output_dir()
        self.type = "output"

    def save_video(
        self,
        frames: torch.Tensor,
        filename_prefix: str,
        fps: int,
        format: str,
        quality: int = 8,
        optimize_gif: bool = True,
    ) -> Dict:
        """
        Save frames as video file.

        Args:
            frames: Frame batch [N, H, W, 3] (ComfyUI IMAGE format)
            filename_prefix: Filename prefix
            fps: Frames per second
            format: Output format (mp4, gif, webm)
            quality: Video quality (1-10)
            optimize_gif: Whether to optimize GIF

        Returns:
            Dictionary with UI output information
        """
        # Runtime validation of frames tensor
        if not isinstance(frames, torch.Tensor):
            raise ValueError(f"frames must be a torch.Tensor, got {type(frames)}")

        if frames.dim() != 4:
            raise ValueError(f"frames must be 4D tensor [N,H,W,C], got shape {frames.shape}")

        if frames.shape[-1] != 3:
            raise ValueError(f"frames must have 3 channels (RGB), got {frames.shape[-1]} channels")

        if frames.shape[0] < 1:
            raise ValueError(f"frames must contain at least 1 frame, got {frames.shape[0]} frames")

        print(f"\n{'='*60}")
        print(f"Saving TurboDiffusion Video")
        print(f"{'='*60}")
        print(f"Frames shape: {frames.shape}")
        print(f"FPS: {fps}")
        print(f"Format: {format}")
        print(f"Quality: {quality}")
        print(f"{'='*60}\n")

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}"

        # Full output path (extension will be added by save_video)
        output_path = self.output_dir / filename

        try:
            # Format-specific kwargs
            kwargs = {}
            if format == "mp4" or format == "webm":
                kwargs["quality"] = quality
            elif format == "gif":
                kwargs["optimize"] = optimize_gif
                kwargs["loop"] = 0  # Infinite loop

            # Save video
            save_video(
                frames=frames,
                output_path=str(output_path),
                fps=fps,
                format=format,
                **kwargs,
            )

            # Get the actual saved file path
            saved_file = output_path.with_suffix(f".{format}")

            print(f"{'='*60}")
            print(f"Video saved successfully!")
            print(f"Location: {saved_file}")
            print(f"Size: {self._get_file_size(saved_file)}")
            print(f"{'='*60}\n")

            # Return UI information
            return {
                "ui": {
                    "videos": [
                        {
                            "filename": saved_file.name,
                            "subfolder": "",
                            "type": "output",
                        }
                    ]
                }
            }

        except ImportError as e:
            error_msg = (
                f"\n{'='*60}\n"
                f"ERROR: Missing video export dependency!\n"
                f"{'='*60}\n"
                f"{str(e)}\n\n"
                f"To enable video export, install optional dependencies:\n"
                f"  uv sync --extra video\n"
                f"OR:\n"
                f"  pip install opencv-python imageio imageio-ffmpeg\n"
                f"{'='*60}\n"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = (
                f"\n{'='*60}\n"
                f"ERROR: Failed to save video!\n"
                f"{'='*60}\n"
                f"{str(e)}\n"
                f"{'='*60}\n"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e

    def _get_output_dir(self) -> Path:
        """
        Get the output directory for saved videos.

        Returns:
            Path to output directory
        """
        # Try to use ComfyUI's output directory if available
        try:
            import folder_paths
            output_dir = Path(folder_paths.get_output_directory())
            return output_dir / "turbodiffusion_videos"
        except ImportError:
            # Fallback to local output directory
            current_dir = Path(__file__).parent.parent
            return current_dir / "output"

    def _get_file_size(self, filepath: Path) -> str:
        """
        Get human-readable file size.

        Args:
            filepath: Path to file

        Returns:
            Formatted file size string
        """
        if not filepath.exists():
            return "0 B"

        size_bytes = filepath.stat().st_size

        # Convert to human-readable format
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0

        return f"{size_bytes:.2f} TB"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        Determine if the node needs to be re-executed.
        Always re-execute output nodes.
        """
        # Always re-execute (use timestamp to ensure uniqueness)
        return datetime.now().isoformat()

    @classmethod
    def VALIDATE_INPUTS(cls, frames, filename_prefix, fps, format, **kwargs):
        """Validate input parameters.

        Note: frames is a link reference at validation time, not the actual tensor.
        Only validate literal input parameters here.
        """
        # Validate filename_prefix (literal input)
        if not isinstance(filename_prefix, str) or not filename_prefix.strip():
            return "filename_prefix must be a non-empty string"

        # Validate fps (literal input)
        if not isinstance(fps, int) or not (1 <= fps <= 60):
            return f"fps must be an integer between 1 and 60, got {fps}"

        # Validate format (literal input)
        if format not in ["mp4", "gif", "webm"]:
            return f"format must be one of [mp4, gif, webm], got {format}"

        # Validate quality (optional literal input)
        quality = kwargs.get("quality", 8)
        if not isinstance(quality, int) or not (1 <= quality <= 10):
            return f"quality must be an integer between 1 and 10, got {quality}"

        return True
