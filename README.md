# ComfyUI TurboDiffusion T2V

ComfyUI custom node for [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) Text-to-Video generation using Wan-2.1-T2V-1.3B-480P model.

## Features

- ✅ **Complete T2V Pipeline**: Single node handles text encoding, pure noise initialization, sampling, and decoding
- ✅ **SLA Attention**: 2-3x faster inference with Sparse Linear Attention optimization
- ✅ **Quantized Models**: Supports int8 block-wise quantized .pth models
- ✅ **Single Model Sampling**: Optimized for T2V generation with single model
- ✅ **Memory Management**: Automatic model loading/offloading for efficient VRAM usage
- ✅ **Fully Self-Contained**: All TurboDiffusion code included, no external dependencies
- ✅ **Independent**: Works independently from ComfyUI TurboDiffusion I2V node

## Requirements

- **GPU**: NVIDIA RTX 3090/4090 or better (12GB+ VRAM recommended)
- **Software**: 
  - Python >= 3.9
  - PyTorch >= 2.0
  - ComfyUI
  - CUDA-capable GPU

## Installation

1. Navigate to ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes/
```

2. Clone this repository:
```bash
git clone https://github.com/trandangtrungduc/Comfyui-TurboDiffusion-T2V.git
```

3. Install dependencies (optional, for video export):
```bash
cd Comfyui-TurboDiffusion-T2V
pip install opencv-python imageio imageio-ffmpeg
```

4. Restart ComfyUI

## Required Models

Download and place in your ComfyUI models directories:

### 1. Diffusion Model (`ComfyUI/models/diffusion_models/`)
- `TurboWan2.1-T2V-1.3B-480P-quant.pth`

Download from: https://huggingface.co/thu-ml/TurboWan2.1-T2V-1.3B-480P

### 2. VAE (`ComfyUI/models/vae/`)
- `wan_2.1_vae.safetensors` (or `.pth`)

Download from: https://huggingface.co/thu-ml/TurboWan2.1-T2V-1.3B-480P

### 3. Text Encoder (`ComfyUI/models/diffusion_models/` or `checkpoints/`)
- `models_t5_umt5-xxl-enc-bf16.pth`

Download from: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B

## Quick Start

### Basic Workflow

The workflow uses 4 main nodes:

1. **TurboWanModelLoader** → Load T2V model (.pth with SLA attention)
2. **TurboWanT5Loader** → Load umT5-xxl text encoder
3. **TurboWanVAELoader** → Load Wan2.1 VAE
4. **TurboDiffusionT2VSampler** → Complete T2V inference (generates video frames)
5. **TurboDiffusionSaveVideo** → Save as MP4/GIF/WebM (optional)

See `turbowan_t2v_workflow.json` for a complete workflow example.

### Example Usage

1. Load the diffusion model with `TurboWanModelLoader`
   - Select `TurboWan2.1-T2V-1.3B-480P-quant.pth`
   - Set `attention_type` to `"sagesla"` (or `"sla"` if SpargeAttn not available)
   - Set `offload_mode` to `"comfy_native"` for best performance

2. Load the text encoder with `TurboWanT5Loader`
   - Select `models_t5_umt5-xxl-enc-bf16.pth`

3. Load the VAE with `TurboWanVAELoader`
   - Select `wan_2.1_vae.safetensors`

4. Generate video with `TurboDiffusionT2VSampler`
   - Connect model, text_encoder, and vae
   - Enter your text prompt
   - Set `num_frames` (default: 81, must be 8n+1)
   - Set `num_steps` (1-4, default: 4)
   - Choose `resolution` and `aspect_ratio`

5. Save video with `TurboDiffusionSaveVideo` (optional)
   - Connect frames output
   - Set filename prefix, fps, and format

## Node Reference

### TurboWanModelLoader

Loads quantized .pth TurboDiffusion models with SLA attention optimization.

**Inputs**:
- `model_name` (required): Model file from diffusion_models/
- `attention_type` (optional): `"sagesla"` (recommended, requires SpargeAttn), `"sla"`, or `"original"` (default: `"sla"`)
- `sla_topk` (optional): Top-k ratio for sparse attention (default: 0.1)
- `offload_mode` (optional): `"comfy_native"` (recommended), `"layerwise_gpu"`, or `"cpu_only"` (default: `"comfy_native"`)

**Outputs**:
- `MODEL`: Loaded TurboDiffusion model (lazy-loaded, loads on first use)

**Notes**:
- Model architecture is auto-detected from filename
- Models are loaded to CPU initially and moved to GPU during inference
- Lazy loading eliminates upfront loading time in workflows

### TurboWanT5Loader

Loads umT5-XXL text encoder for text prompt encoding.

**Inputs**:
- `encoder_name` (required): T5 encoder checkpoint filename

**Outputs**:
- `TEXT_ENCODER`: Config dict with `t5_path` key

**Notes**:
- Searches in `diffusion_models/` and `checkpoints/` directories
- Returns a config dict, not the actual model (model loads on-demand)

### TurboWanVAELoader

Loads Wan2.1 VAE with video encoding/decoding support.

**Inputs**:
- `vae_name` (required): VAE file from models/vae/ folder

**Outputs**:
- `VAE`: Wan2pt1VAEInterface object with temporal support (lazy-loaded)

**Notes**:
- This is NOT the same as ComfyUI's standard VAELoader
- The Wan VAE handles video frames (B, C, T, H, W) with temporal compression
- VAE is wrapped with ComfyUI-compatible device management
- Automatically moves to GPU for decoding, then returns to CPU

### TurboDiffusionT2VSampler

Complete T2V inference with single model sampling.

**Inputs** (Required):
- `model`: Single model for T2V generation (from TurboWanModelLoader)
- `prompt`: Text prompt for video generation (required if not using conditioning)
- `vae`: VAE from TurboWanVAELoader
- `num_frames`: Frames to generate (must be 8n+1, default: 81, range: 9-241)
- `num_steps`: Sampling steps (1-4, default: 4)
- `resolution`: `"480"`, `"480p"`, `"512"`, `"720"`, `"720p"` (default: `"480p"`)
- `aspect_ratio`: `"16:9"`, `"9:16"`, `"4:3"`, `"3:4"`, `"1:1"` (default: `"16:9"`)
- `sigma_max`: Initial sigma for rCM (default: 80.0 for T2V)
- `seed`: Random seed (default: 0)
- `use_ode`: ODE vs SDE sampling (default: false = SDE)

**Inputs** (Optional):
- `conditioning`: Text conditioning from CLIPTextEncode (alternative to prompt)
- `text_encoder`: T5 encoder config (required if using prompt string)

**Outputs**:
- `frames`: Generated video frames (B*T, H, W, C) in ComfyUI IMAGE format

**Resolution Guide**:
- `"480"`: 480×480 (1:1), 640×480 (4:3), etc. - **Lower VRAM usage**
- `"480p"`: 640×640 (1:1), 832×480 (16:9), etc. - Higher VRAM usage
- `"720"` / `"720p"`: Higher resolutions for high VRAM GPUs

**VRAM Recommendations**:
- **Low VRAM (8-12GB)**: Use `"480"` with 49 frames
- **Medium VRAM (16GB)**: Use `"480p"` with 81 frames
- **High VRAM (24GB+)**: Use `"720p"` with 81+ frames

**How it works**:
1. Encodes text prompt with T5 (or extracts from conditioning)
2. Initializes pure random noise (no image input)
3. Creates conditioning dict with only `crossattn_emb` (text embedding)
4. Runs rCM sampling with single model
5. Decodes final latents with VAE
6. Returns frames in ComfyUI IMAGE format

### TurboDiffusionSaveVideo

Saves frame sequence as video file.

**Inputs** (Required):
- `frames`: Video frames from sampler (IMAGE tensor)
- `filename_prefix`: Output filename prefix
- `fps`: Frames per second (1-60, default: 24)
- `format`: `"mp4"`, `"gif"`, or `"webm"` (default: `"mp4"`)

**Inputs** (Optional):
- `quality`: Compression quality (1-10, default: 8, only for MP4/WebM)
- `optimize_gif`: Whether to optimize GIF file size (default: true, only for GIF)

**Outputs**:
- None (output node, saves to `ComfyUI/output/turbodiffusion_videos/`)

**Notes**:
- Requires `opencv-python`, `imageio`, and `imageio-ffmpeg` for video export
- Files are saved with timestamp: `{filename_prefix}_{YYYYMMDD_HHMMSS}.{format}`

## Performance

### Benchmarks (RTX 3090, SageSLA attention)

- **480p, 81 frames, 4 steps**: ~60-90 seconds
- **480, 49 frames, 4 steps**: ~30-50 seconds
- **2-3x faster** than original attention
- **~12-15GB VRAM** usage with automatic offloading

### Optimization Tips

1. **Use SageSLA attention**: 2-3x speedup over original attention
2. **Reduce frame count**: Use 49 frames instead of 81 for faster generation
3. **Lower resolution**: Use `"480"` instead of `"480p"` for lower VRAM
4. **Fewer steps**: Use 2-3 steps instead of 4 for faster generation (slight quality trade-off)
5. **ComfyUI native offloading**: Best performance with `offload_mode="comfy_native"`

## Technical Details

### Architecture

- **Model**: TurboDiffusion Wan2.1-T2V-1.3B (text-to-video, 1.3B parameters)
- **Quantization**: Block-wise int8 with automatic dequantization
- **Attention**: SageSLA (Sparse Linear Attention) for 2-3x speedup
- **Sampling**: rCM (Rectified Consistency Model) with single model
- **VAE**: Wan2.1 VAE (16 channel latents, temporal compression)
- **Text Encoder**: umT5-xxl

### Memory Management

**ComfyUI Integration**:
- VAE wrapped with ComfyUI-compatible device management
- Automatic loading/offloading integrated with ComfyUI's model management system
- Calls `comfy.model_management.soft_empty_cache()` before sampling
- VAE automatically moves to GPU for decoding, then returns to CPU

**Model Offloading**:
- Diffusion models start on CPU
- Models moved to GPU only during sampling
- Automatic offloading after sampling completes
- Text embeddings kept on GPU after encoding

**Offload Modes**:
- `comfy_native`: Uses ComfyUI's native async weight offloading (pinned RAM, 2 streams) - **Recommended**
- `layerwise_gpu`: Swaps blocks to GPU just-in-time (ComfyUI-style)
- `cpu_only`: Runs the whole forward on CPU (very slow, only for debugging)

### Differences from I2V

- **No image input**: Starts from pure random noise (not image-conditioned)
- **Text-only conditioning**: Only uses `crossattn_emb`, no `y_B_C_T_H_W` image conditioning
- **Default values**: `sigma_max=80`, `num_frames=81` (matching T2V script defaults)
- **Direct prompt support**: Can use prompt string directly (requires T5 loader)
- **Single model only**: T2V uses single model sampling (no dual-expert support)

## Troubleshooting

### Common Issues

**"ModuleNotFoundError" or "Could not import TurboDiffusion modules"**:
- Restart ComfyUI after installation
- Ensure all files in `turbodiffusion_vendor/` are present
- Check that Python path includes the custom_nodes directory

**"Model not found"**:
- Verify model files are in correct ComfyUI directories:
  - Diffusion model: `ComfyUI/models/diffusion_models/`
  - VAE: `ComfyUI/models/vae/`
  - T5 encoder: `ComfyUI/models/diffusion_models/` or `ComfyUI/models/checkpoints/`
- Check file names match exactly (case-sensitive)

**CUDA Out of Memory (OOM)**:
- Reduce resolution: Use `"480"` instead of `"480p"`
- Reduce frame count: Use 49 instead of 81 frames
- Use fewer steps: Try 2-3 steps instead of 4
- Enable model offloading: Set `offload_mode="comfy_native"`

**Slow performance**:
- Check that `attention_type` is `"sla"` or `"sagesla"` (not `"original"`)
- Install SpargeAttn for SageSLA: `pip install git+https://github.com/thu-ml/SpargeAttn.git`
- Ensure GPU is being used (check CUDA availability in logs)

**"TurboDiffusionT2VSampler" missing from node list**:
- Ensure all vendored files were copied (`turbodiffusion_vendor/` directory)
- Restart ComfyUI
- Check for import errors in ComfyUI console

**"text_encoder is required" error**:
- Provide `text_encoder` parameter when using `prompt` string
- OR use `conditioning` input instead of `prompt` (from CLIPTextEncode)

**Video export fails**:
- Install video dependencies: `pip install opencv-python imageio imageio-ffmpeg`
- Check that output directory is writable
- Verify frames tensor is valid (4D: [N, H, W, 3])

### Debugging

Enable verbose logging by checking ComfyUI console output. The nodes use `TimedLogger` which provides detailed timing and memory information.

## Workflow Examples

### Basic T2V Generation

```
TurboWanModelLoader → TurboDiffusionT2VSampler
TurboWanT5Loader    → TurboDiffusionT2VSampler
TurboWanVAELoader   → TurboDiffusionT2VSampler
                     → TurboDiffusionSaveVideo
```

### Using Conditioning Instead of Prompt

```
CLIPTextEncode → TurboDiffusionT2VSampler (conditioning input)
TurboWanModelLoader → TurboDiffusionT2VSampler
TurboWanVAELoader   → TurboDiffusionT2VSampler
```

## Acknowledgements

This project inherits code from the following open-source projects:

- **[TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)** by THU-ML - Core diffusion model architecture, inference pipeline, and sampling algorithms (vendored in `turbodiffusion_vendor/`)
- **[ComfyUI TurboDiffusion](https://github.com/anveshane/Comfyui_turbodiffusion)** by anveshane - ComfyUI integration patterns and memory management approaches

## License

Apache 2.0 (same as TurboDiffusion)

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## Support

For issues and questions:
- GitHub Issues: https://github.com/trandangtrungduc/Comfyui-TurboDiffusion-T2V/issues
- ComfyUI Discord: [ComfyUI Community](https://discord.gg/comfyui)
