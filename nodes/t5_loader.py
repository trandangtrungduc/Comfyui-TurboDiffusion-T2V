"""TurboDiffusion T5 Text Encoder Loader node for ComfyUI."""

from typing import Tuple
from pathlib import Path

from ..utils.model_management import get_checkpoint_dir


class TurboWanT5Loader:
    """
    ComfyUI node for loading umT5-XXL text encoder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node."""
        return {
            "required": {
                "encoder_name": (
                    cls._get_encoder_list(),
                    {
                        "default": "models_t5_umt5-xxl-enc-bf16.pth",
                        "tooltip": "Select the T5 text encoder checkpoint to load",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TEXT_ENCODER",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "load_encoder"
    CATEGORY = "video/turbodiffusion"
    DESCRIPTION = "Load umT5-XXL text encoder for TurboDiffusion text conditioning"

    @classmethod
    def _get_encoder_list(cls):
        """Get list of available T5 encoder checkpoints."""
        encoder_files = []

        # Search in diffusion_models folder
        try:
            import folder_paths
            for search_dir in folder_paths.get_folder_paths("diffusion_models"):
                search_path = Path(search_dir)
                if search_path.exists():
                    for enc_file in search_path.glob("*t5*.pth"):
                        encoder_files.append(enc_file.name)
        except:
            pass

        # Search in local checkpoints
        checkpoint_dir = get_checkpoint_dir()
        if checkpoint_dir.exists():
            for enc_file in checkpoint_dir.glob("*t5*.pth"):
                if enc_file.name not in encoder_files:
                    encoder_files.append(enc_file.name)

        # Default if none found
        if not encoder_files:
            encoder_files = ["models_t5_umt5-xxl-enc-bf16.pth"]

        return sorted(encoder_files)

    def load_encoder(self, encoder_name: str) -> Tuple[dict]:
        """
        Load the T5 text encoder checkpoint.

        Args:
            encoder_name: Name of T5 encoder checkpoint file

        Returns:
            Tuple containing text encoder configuration dictionary
        """
        print(f"\n{'='*60}")
        print(f"Loading TurboWan T5 Text Encoder")
        print(f"{'='*60}")
        print(f"Encoder: {encoder_name}")

        # Find encoder file
        encoder_path = None

        # Search in diffusion_models folder
        try:
            import folder_paths
            for search_dir in folder_paths.get_folder_paths("diffusion_models"):
                search_path = Path(search_dir) / encoder_name
                if search_path.exists():
                    encoder_path = search_path
                    break
        except:
            pass

        # Search in local checkpoints
        if encoder_path is None:
            checkpoint_dir = get_checkpoint_dir()
            local_enc = checkpoint_dir / encoder_name
            if local_enc.exists():
                encoder_path = local_enc

        if encoder_path is None:
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"ERROR: T5 Encoder not found: {encoder_name}\n"
                f"{'='*60}\n"
                f"Please download from:\n"
                f"https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B\n"
                f"\nPlace in: ComfyUI/models/diffusion_models/\n"
                f"{'='*60}\n"
            )

        encoder_config = {
            "t5_path": str(encoder_path),
            "encoder_name": encoder_name,
        }

        print(f"T5 Encoder loaded from: {encoder_path}")
        print(f"{'='*60}\n")

        return (encoder_config,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache based on encoder name."""
        return kwargs.get("encoder_name", "")

    @classmethod
    def VALIDATE_INPUTS(cls, encoder_name):
        """Validate input parameters."""
        if not isinstance(encoder_name, str) or not encoder_name:
            return "encoder_name must be a non-empty string"
        return True
