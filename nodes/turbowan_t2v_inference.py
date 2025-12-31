"""
TurboDiffusion T2V Inference Node

Complete inference pipeline for TurboDiffusion Text-to-Video generation.
This node handles the full sampling process internally, starting from pure noise.
"""

import torch
from typing import Tuple, Optional
from einops import repeat

import comfy.model_management

try:
    from ..utils.timing import TimedLogger
    from ..turbodiffusion_vendor.rcm.datasets.utils import VIDEO_RES_SIZE_INFO
    from ..turbodiffusion_vendor.rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    from ..turbodiffusion_vendor.rcm.cm_sampler import rcm_sampler
    from ..turbodiffusion_vendor.rcm.utils.umt5 import get_umt5_embedding, clear_umt5_memory
    TURBODIFFUSION_AVAILABLE = True
except ImportError as e:
    TURBODIFFUSION_AVAILABLE = False
    print(f"ERROR: Could not import TurboDiffusion modules: {e}")


class TurboDiffusionT2VSampler:
    """
    Complete TurboDiffusion T2V inference node with single model sampling.

    This node handles the entire inference pipeline:
    - Text encoding (direct prompt or from conditioning)
    - Pure random noise initialization
    - Single model rCM sampling
    - VAE decoding
    - Automatic memory management
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "TurboDiffusion T2V model from TurboWanModelLoader"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt for video generation"
                }),
                "vae": ("VAE", {"tooltip": "Wan2.1 VAE from VAELoader"}),
                "num_frames": ("INT", {
                    "default": 81,
                    "min": 9,
                    "max": 241,
                    "step": 8,
                    "tooltip": "Number of frames to generate (default: 81 for T2V)"
                }),
                "num_steps": ([1, 2, 3, 4], {
                    "default": 4,
                    "tooltip": "Number of sampling steps (1-4 for distilled model)"
                }),
                "resolution": (["480", "480p", "512", "720", "720p"], {
                    "default": "480p",
                    "tooltip": "Base resolution (480=480x480, 480p=640x640 for 1:1)"
                }),
                "aspect_ratio": (["16:9", "9:16", "4:3", "3:4", "1:1"], {
                    "default": "16:9",
                    "tooltip": "Aspect ratio"
                }),
                "sigma_max": ("FLOAT", {
                    "default": 80.0,
                    "min": 1.0,
                    "max": 1000.0,
                    "tooltip": "Initial sigma for rCM sampling (default: 80 for T2V)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed"
                }),
                "use_ode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use ODE sampling (sharper but less robust)"
                }),
            },
            "optional": {
                "conditioning": ("CONDITIONING", {"tooltip": "Text conditioning from CLIPTextEncode (alternative to prompt)"}),
                "text_encoder": ("TEXT_ENCODER", {"tooltip": "T5 encoder config (required if using prompt string)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "video/turbodiffusion"
    DESCRIPTION = "Complete TurboDiffusion T2V inference with single model sampling"

    def generate(
        self,
        model,
        prompt: str,
        vae,
        num_frames: int,
        num_steps: int,
        resolution: str,
        aspect_ratio: str,
        sigma_max: float,
        seed: int,
        use_ode: bool,
        conditioning: Optional = None,
        text_encoder: Optional[dict] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Run complete TurboDiffusion T2V inference.

        Returns:
            Tuple containing generated video frames as IMAGE tensor
        """
        if not TURBODIFFUSION_AVAILABLE:
            raise RuntimeError("TurboDiffusion modules not available!")

        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16

        logger = TimedLogger("T2V-Inference")
        logger.section("TurboDiffusion T2V Inference")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.log(f"CUDA available: Yes ({gpu_count} GPU(s) detected)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                total_memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.log(f"  GPU {i}: {gpu_name} - {total_memory_gb:.2f}GB total VRAM")

            current_device = device if isinstance(device, int) else 0
            allocated_gb = torch.cuda.memory_allocated(current_device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(current_device) / (1024**3)
            total_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            free_gb = total_gb - reserved_gb
            logger.log(f"VRAM at start: {allocated_gb:.2f}GB used, {free_gb:.2f}GB free (of {total_gb:.2f}GB total)")
        else:
            logger.log(f"CUDA available: No (running on CPU)")

        logger.log("Using single model setup (T2V only supports single model)")

        logger.log(f"Frames: {num_frames}, Steps: {num_steps}, Resolution: {resolution} {aspect_ratio}")
        logger.log(f"Sigma: {sigma_max}, Seed: {seed}")

        tokenizer = vae

        w, h = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
        lat_h = h // tokenizer.spatial_compression_factor
        lat_w = w // tokenizer.spatial_compression_factor
        lat_t = tokenizer.get_latent_num_frames(num_frames)

        frame_tensor_gb = (num_frames * 3 * h * w * 4) / (1024**3)
        logger.log(f"Target resolution: {w}x{h}, Latent shape: {lat_t}x{lat_h}x{lat_w}")
        logger.log(f"Frame tensor size: ~{frame_tensor_gb:.2f}GB")

        if frame_tensor_gb > 1.5:
            logger.log(f"⚠️  WARNING: Large frame tensor ({frame_tensor_gb:.2f}GB) may cause OOM!")
            logger.log(f"   Consider: resolution='480' (not '480p'), or fewer frames (e.g., 49 instead of {num_frames})")

        if prompt and prompt.strip():
            logger.log("Encoding text prompt with T5...")
            if text_encoder is None:
                raise ValueError(
                    "text_encoder is required when using prompt string. "
                    "Either provide text_encoder parameter or use conditioning input instead."
                )
            t5_path = text_encoder.get("t5_path")
            if not t5_path:
                raise ValueError("text_encoder must contain 't5_path' key")
            
            with torch.no_grad():
                text_emb = get_umt5_embedding(checkpoint_path=t5_path, prompts=prompt).to(device=device, dtype=dtype)
            clear_umt5_memory()
            logger.log(f"Text embedding shape: {text_emb.shape}")
        elif conditioning is not None:
            logger.log("Extracting text embedding from conditioning...")
            text_emb = conditioning[0][0]

            if isinstance(text_emb, dict):
                if "crossattn_emb" in text_emb:
                    text_emb = text_emb["crossattn_emb"]
                elif "pooled_output" in text_emb:
                    text_emb = text_emb["pooled_output"]
                else:
                    for key, val in text_emb.items():
                        if isinstance(val, torch.Tensor):
                            text_emb = val
                            break

            logger.log(f"Text embedding shape: {text_emb.shape}")
            text_emb = text_emb.to(device=device, dtype=dtype)
        else:
            raise ValueError("Either 'prompt' or 'conditioning' must be provided")

        logger.log(f"Initializing noise with seed {seed}...")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]
        init_noise = torch.randn(
            1, *state_shape,
            dtype=torch.float32,
            device=device,
            generator=generator
        )

        condition = {
            "crossattn_emb": text_emb,
        }

        logger.log("Running rCM sampling...")

        # comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # import gc
        # gc.collect()
        # torch.cuda.empty_cache()

        if torch.cuda.is_available():
            current_device = device if isinstance(device, int) else 0
            allocated_gb = torch.cuda.memory_allocated(current_device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(current_device) / (1024**3)
            total_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            free_gb = total_gb - reserved_gb
            logger.log(f"VRAM before model: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved, {free_gb:.2f}GB free")

        logger.log("Loading model...")
        model = model.to(device)

        if torch.cuda.is_available():
            current_device = device if isinstance(device, int) else 0
            allocated_gb = torch.cuda.memory_allocated(current_device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(current_device) / (1024**3)
            total_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            free_gb = total_gb - reserved_gb
            logger.log(f"VRAM after model: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved, {free_gb:.2f}GB free")

        logger.log(f"Sampling with model (all {num_steps} steps)...")
        with torch.no_grad():
            x = rcm_sampler(
                model,
                init_noise,
                condition,
                num_steps=num_steps,
                sigma_max=sigma_max,
                use_ode=use_ode,
                generator=generator,
                start_step=0,
                end_step=num_steps,
                verbose=True
            )

        logger.log("Offloading model...")
        model = model.cpu()
        torch.cuda.empty_cache()

        logger.log("Decoding latents with VAE...")

        with torch.no_grad():
            decoded_frames = tokenizer.decode(x)

        logger.log("VAE decoding complete (VAE automatically offloaded to CPU)")
        torch.cuda.empty_cache()

        decoded_frames = decoded_frames.permute(0, 2, 3, 4, 1).contiguous()
        out_h = decoded_frames.shape[2]
        out_w = decoded_frames.shape[3]
        out_c = decoded_frames.shape[4]
        decoded_frames = decoded_frames.reshape(-1, out_h, out_w, out_c)

        decoded_frames = (decoded_frames + 1.0) / 2.0
        decoded_frames = decoded_frames.clamp(0, 1)

        logger.log(f"✓ Successfully generated {decoded_frames.shape[0]} frames!")
        logger.log(f"Total inference time: {logger.elapsed():.2f}s")
        print(f"{'='*60}\n")

        return (decoded_frames.cpu(),)

