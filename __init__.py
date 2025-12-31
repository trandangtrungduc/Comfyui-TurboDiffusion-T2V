"""
ComfyUI TurboDiffusion T2V Custom Node

This package provides nodes for TurboDiffusion Text-to-Video generation using
Wan-2.1-T2V-1.3B model.

Nodes:
- TurboDiffusionT2VSampler: Complete T2V inference with single or dual-expert sampling
- TurboWanModelLoader: Load TurboDiffusion models
- TurboWanVAELoader: Load Wan2.1 VAE
- TurboWanT5Loader: Load umT5-XXL text encoder
- TurboDiffusionSaveVideo: Save generated videos

Usage:
1. Load model: Use TurboWanModelLoader or standard ComfyUI model loader
2. Load VAE: Use TurboWanVAELoader or standard ComfyUI VAE loader
3. Load T5 encoder: Use TurboWanT5Loader or provide T5 path (if using prompt string)
4. Generate: TurboDiffusionT2VSampler → returns video frames
5. Save: Use TurboDiffusionSaveVideo or standard ComfyUI video save nodes

Note: This node is fully self-contained with all dependencies included.

Repository: https://github.com/thu-ml/TurboDiffusion
License: Apache 2.0
"""

try:
    from .nodes.turbowan_t2v_inference import TurboDiffusionT2VSampler
    from .nodes.turbowan_model_loader import TurboWanModelLoader as TurboWanT2VModelLoader
    from .nodes.vae_loader import TurboWanVAELoader as TurboWanT2VVAELoader
    from .nodes.t5_loader import TurboWanT5Loader as TurboWanT2VT5Loader
    from .nodes.video_saver import TurboDiffusionSaveVideo as TurboDiffusionT2VSaveVideo

    NODE_CLASS_MAPPINGS = {
        "TurboDiffusionT2VSampler": TurboDiffusionT2VSampler,
        "TurboWanT2VModelLoader": TurboWanT2VModelLoader,
        "TurboWanT2VVAELoader": TurboWanT2VVAELoader,
        "TurboWanT2VT5Loader": TurboWanT2VT5Loader,
        "TurboDiffusionT2VSaveVideo": TurboDiffusionT2VSaveVideo,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "TurboDiffusionT2VSampler": "TurboDiffusion T2V Sampler",
        "TurboWanT2VModelLoader": "TurboWan T2V Model Loader",
        "TurboWanT2VVAELoader": "TurboWan T2V VAE Loader",
        "TurboWanT2VT5Loader": "TurboWan T2V T5 Loader",
        "TurboDiffusionT2VSaveVideo": "Save Video (T2V)",
    }
except Exception as e:
    print(f"ERROR: Failed to load TurboDiffusion T2V nodes: {e}")
    import traceback
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "0.1.0"
__author__ = "ComfyUI TurboDiffusion T2V Contributors"

print("\n" + "=" * 60)
print("ComfyUI TurboDiffusion T2V Node")
print("=" * 60)
print(f"Version: {__version__}")
print(f"Loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  - {display_name} ({node_name})")
print("\nFeatures:")
print("  - Text-to-Video generation with Wan-2.1-T2V-1.3B-480P")
print("  - Single model or dual-expert support")
print("  - Direct text prompt input or conditioning input")
print("  - Automatic memory management")
print("  - Fully self-contained (no external dependencies)")
print("  - All TurboDiffusion code included")
print("  - Works independently from Comfyui_turbodiffusion")
print("\nRequires:")
print("  - ComfyUI")
print("  - Wan-2.1-T2V-1.3B-480P model from HuggingFace")
print("=" * 60 + "\n")

