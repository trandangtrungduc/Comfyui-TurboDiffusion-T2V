"""Model management utilities for TurboDiffusion."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm


# Model configuration mappings
MODEL_CONFIGS = {
    "A14B-high-720p": {
        "repo": "TurboDiffusion/TurboWan2.2-I2V-A14B-720P",
        "filename": "TurboWan2.2-I2V-A14B-high-720P.pth",
        "size": "28.6GB",
        "quantized": False,
    },
    "A14B-high-720p-quant": {
        "repo": "TurboDiffusion/TurboWan2.2-I2V-A14B-720P",
        "filename": "TurboWan2.2-I2V-A14B-high-720P-quant.pth",
        "size": "14.5GB",
        "quantized": True,
    },
    "A14B-low-720p": {
        "repo": "TurboDiffusion/TurboWan2.2-I2V-A14B-720P",
        "filename": "TurboWan2.2-I2V-A14B-low-720P.pth",
        "size": "28.6GB",
        "quantized": False,
    },
    "A14B-low-720p-quant": {
        "repo": "TurboDiffusion/TurboWan2.2-I2V-A14B-720P",
        "filename": "TurboWan2.2-I2V-A14B-low-720P-quant.pth",
        "size": "14.5GB",
        "quantized": True,
    },
}

VAE_CONFIG = {
    "repo": "Wan-AI/Wan2.1-T2V-1.3B",
    "filename": "wan_2.1_vae.safetensors",
}

T5_CONFIG = {
    "repo": "Wan-AI/Wan2.1-T2V-1.3B",
    "filename": "models_t5_umt5-xxl-enc-bf16.pth",
}


def get_checkpoint_dir() -> Path:
    """
    Get the checkpoint directory path.

    Checks in this order:
    1. ComfyUI/models/diffusion_models/ (standard ComfyUI diffusion models location)
    2. custom_nodes/comfyui-turbodiffusion/checkpoints/ (node's local folder)

    Returns:
        Path to checkpoint directory
    """
    # Try to use ComfyUI's standard diffusion_models folder first
    try:
        import folder_paths
        # Use folder_paths to get the standard diffusion_models path
        diffusion_models_dir = folder_paths.get_folder_paths("diffusion_models")

        if diffusion_models_dir and len(diffusion_models_dir) > 0:
            # Use the first diffusion_models directory
            return Path(diffusion_models_dir[0])
    except (ImportError, AttributeError, KeyError):
        # folder_paths not available or diffusion_models not registered
        pass

    # Try alternative: ComfyUI/models/diffusion_models
    try:
        import folder_paths
        comfyui_models = Path(folder_paths.models_dir)
        diffusion_models = comfyui_models / "diffusion_models"

        if diffusion_models.exists():
            return diffusion_models

        # Create it if ComfyUI models dir exists
        diffusion_models.mkdir(exist_ok=True, parents=True)
        return diffusion_models
    except (ImportError, AttributeError):
        pass

    # Fallback: use local checkpoints directory
    current_dir = Path(__file__).parent.parent
    checkpoint_dir = current_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


def get_checkpoint_path(variant: str, resolution: str = "720p") -> Optional[Path]:
    """
    Get the local path to a checkpoint file if it exists.

    Searches in multiple locations:
    1. ComfyUI/models/turbodiffusion/
    2. custom_nodes/comfyui-turbodiffusion/checkpoints/

    Args:
        variant: Model variant (e.g., "A14B-high", "A14B-high-quant")
        resolution: Resolution (480p or 720p)

    Returns:
        Path to checkpoint file if it exists, None otherwise
    """
    model_key = f"{variant}-{resolution}"

    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model variant: {model_key}. "
            f"Available variants: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_key]
    filename = config["filename"]

    # Search locations in order of preference
    search_paths = []

    # 1. ComfyUI standard diffusion_models folder
    try:
        import folder_paths
        diffusion_models_dirs = folder_paths.get_folder_paths("diffusion_models")
        if diffusion_models_dirs:
            for dir_path in diffusion_models_dirs:
                search_paths.append(Path(dir_path))
    except (ImportError, AttributeError, KeyError):
        pass

    # 2. Alternative: ComfyUI/models/diffusion_models
    try:
        import folder_paths
        comfyui_models = Path(folder_paths.models_dir) / "diffusion_models"
        if comfyui_models.exists():
            search_paths.append(comfyui_models)
    except (ImportError, AttributeError):
        pass

    # 3. Node's local checkpoints folder
    current_dir = Path(__file__).parent.parent
    search_paths.append(current_dir / "checkpoints")

    # Search each location
    for search_dir in search_paths:
        if not search_dir.exists():
            continue

        local_path = search_dir / filename
        if local_path.exists():
            print(f"Found model at: {local_path}")
            return local_path

    return None


def download_model(
    variant: str,
    resolution: str = "720p",
    checkpoint_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download a TurboDiffusion model from HuggingFace.

    Args:
        variant: Model variant (e.g., "A14B-high", "A14B-high-quant")
        resolution: Resolution (480p or 720p)
        checkpoint_dir: Directory to save checkpoint (default: ./checkpoints)
        force: Force download even if file exists

    Returns:
        Path to downloaded checkpoint file

    Raises:
        ValueError: If variant is unknown
    """
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir()
    else:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model_key = f"{variant}-{resolution}"

    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model variant: {model_key}. "
            f"Available variants: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_key]
    local_path = checkpoint_dir / config["filename"]

    # Check if already exists
    if local_path.exists() and not force:
        print(f"Model already exists at: {local_path}")
        return local_path

    print(f"Downloading {config['filename']} ({config['size']})...")
    print(f"From: {config['repo']}")

    try:
        downloaded_path = hf_hub_download(
            repo_id=config["repo"],
            filename=config["filename"],
            local_dir=str(checkpoint_dir),
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded successfully to: {downloaded_path}")
        return Path(downloaded_path)

    except Exception as e:
        raise RuntimeError(
            f"Failed to download model {config['filename']}. "
            f"Error: {e}\n"
            f"Please check your internet connection or download manually from:\n"
            f"https://huggingface.co/{config['repo']}"
        ) from e


def download_vae(checkpoint_dir: Optional[Path] = None, force: bool = False) -> Path:
    """
    Download the Wan2.1 VAE checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoint (default: ./checkpoints)
        force: Force download even if file exists

    Returns:
        Path to VAE checkpoint
    """
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir()
    else:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

    local_path = checkpoint_dir / VAE_CONFIG["filename"]

    if local_path.exists() and not force:
        print(f"VAE already exists at: {local_path}")
        return local_path

    print(f"Downloading {VAE_CONFIG['filename']}...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=VAE_CONFIG["repo"],
            filename=VAE_CONFIG["filename"],
            local_dir=str(checkpoint_dir),
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        print(f"VAE downloaded successfully to: {downloaded_path}")
        return Path(downloaded_path)

    except Exception as e:
        raise RuntimeError(
            f"Failed to download VAE. Error: {e}\n"
            f"Please download manually from:\n"
            f"https://huggingface.co/{VAE_CONFIG['repo']}"
        ) from e


def download_t5_encoder(
    checkpoint_dir: Optional[Path] = None, force: bool = False
) -> Path:
    """
    Download the umT5-XXL text encoder checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoint (default: ./checkpoints)
        force: Force download even if file exists

    Returns:
        Path to T5 encoder checkpoint
    """
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir()
    else:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

    local_path = checkpoint_dir / T5_CONFIG["filename"]

    if local_path.exists() and not force:
        print(f"T5 encoder already exists at: {local_path}")
        return local_path

    print(f"Downloading {T5_CONFIG['filename']}...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=T5_CONFIG["repo"],
            filename=T5_CONFIG["filename"],
            local_dir=str(checkpoint_dir),
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        print(f"T5 encoder downloaded successfully to: {downloaded_path}")
        return Path(downloaded_path)

    except Exception as e:
        raise RuntimeError(
            f"Failed to download T5 encoder. Error: {e}\n"
            f"Please download manually from:\n"
            f"https://huggingface.co/{T5_CONFIG['repo']}"
        ) from e


def load_turbodiffusion_model(
    variant: str,
    resolution: str = "720p",
    checkpoint_path: Optional[str] = None,
    auto_download: bool = True,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Load a TurboDiffusion model with all required components.

    Args:
        variant: Model variant (e.g., "A14B-high-quant")
        resolution: Resolution (480p or 720p)
        checkpoint_path: Optional path to checkpoint (overrides auto-download)
        auto_download: Whether to auto-download missing models
        device: Device to load model on

    Returns:
        Dictionary containing model, VAE, and text encoder

    Raises:
        FileNotFoundError: If checkpoint not found and auto_download is False
        RuntimeError: If model loading fails
    """
    checkpoint_dir = get_checkpoint_dir()

    # Get main model checkpoint
    if checkpoint_path is not None:
        model_path = Path(checkpoint_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at: {checkpoint_path}"
            )
    else:
        # Check if exists locally
        model_path = get_checkpoint_path(variant, resolution)

        # Download if needed
        if model_path is None:
            if not auto_download:
                raise FileNotFoundError(
                    f"Model checkpoint for {variant}-{resolution} not found. "
                    f"Enable auto_download or manually download from HuggingFace."
                )
            model_path = download_model(variant, resolution, checkpoint_dir)

    # Download VAE if needed
    vae_path = checkpoint_dir / VAE_CONFIG["filename"]
    if not vae_path.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"VAE checkpoint not found at: {vae_path}. "
                f"Enable auto_download or manually download from HuggingFace."
            )
        vae_path = download_vae(checkpoint_dir)

    # Download T5 encoder if needed
    t5_path = checkpoint_dir / T5_CONFIG["filename"]
    if not t5_path.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"T5 encoder not found at: {t5_path}. "
                f"Enable auto_download or manually download from HuggingFace."
            )
        t5_path = download_t5_encoder(checkpoint_dir)

    print(f"Loading TurboDiffusion model from: {model_path}")

    try:
        # Load the checkpoint
        # Note: Actual loading will depend on turbodiffusion package API
        # This is a placeholder structure
        model_state = {
            "model_path": str(model_path),
            "vae_path": str(vae_path),
            "t5_path": str(t5_path),
            "variant": variant,
            "resolution": resolution,
            "device": device,
            # The actual model objects will be loaded by the turbodiffusion package
            # in the i2v_generator node
        }

        print("Model configuration loaded successfully")
        return model_state

    except Exception as e:
        raise RuntimeError(
            f"Failed to load model. Error: {e}\n"
            f"Please check that all checkpoints are valid and compatible."
        ) from e


def clear_cuda_cache():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
