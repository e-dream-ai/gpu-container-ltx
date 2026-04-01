# gpu-container-ltx

RunPod serverless container for LTX 2.3 video generation via ComfyUI.

Based on Jef's workflow — two-pass I2V with LCM sampler, spatial upscaling, and audio generation.

## Models Included

| Model | Size | Source | Purpose |
|-------|------|--------|---------|
| `ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors` | ~11GB | Kijai/LTX2.3_comfy | Distilled transformer (FP8) |
| `gemma_3_12B_it_fpmixed.safetensors` | ~6GB | Comfy-Org/ltx-2 | Text encoder (mixed precision) |
| `ltx-2.3_text_projection_bf16.safetensors` | ~2.3GB | Kijai/LTX2.3_comfy | Text projection for DualCLIPLoader |
| `LTX23_video_vae_bf16.safetensors` | ~1.5GB | Kijai/LTX2.3_comfy | Video VAE |
| `LTX23_audio_vae_bf16.safetensors` | ~365MB | Kijai/LTX2.3_comfy | Audio VAE |
| `ltx-2.3-spatial-upscaler-x2-{1.0,1.1}.safetensors` | ~1GB each | Lightricks/LTX-2.3 | 2x latent upscaler (v1.0 + v1.1 hotfix) |
| `taeltx2_3{,_wide}.safetensors` | ~50MB each | madebyollin/taehv | TAESD for fast preview during render |
| 7x camera control LoRAs | ~100MB each | Lightricks | Dolly in/out/left/right, jib up/down, static |

## Pipeline (from Jef's workflow)

```
Pass 1 — Low-res generation (704x512, 121 frames):
  UNETLoader → Power Lora Loader (camera LoRA @ 0.4)
  DualCLIPLoader → CLIPTextEncode → LTXVConditioning
  LoadImage → LTXVPreprocess → LTXVImgToVideoInplace
  EmptyLTXVLatentVideo + LTXVEmptyLatentAudio → LTXVConcatAVLatent
  → SamplerCustomAdvanced (LCM, 8 steps, LTXVScheduler)
  → LTXVSeparateAVLatent

Pass 2 — Upscaled refinement:
  → LTXVLatentUpsampler (2x spatial)
  → LTXVImgToVideoInplace (re-inject ref image)
  → LTXVConcatAVLatent (recombine with audio)
  → SamplerCustomAdvanced (LCM, 3 steps via ManualSigmas: 0.909, 0.725, 0.422, 0.0)
  → LTXVSeparateAVLatent

Output:
  Video → VAEDecodeTiled (tile=512, overlap=64)
  Audio → LTXVAudioVAEDecode
  → VHS_VideoCombine (H.264 MP4, CRF 19, 24fps)
```

## Custom Nodes

- **ComfyUI-LTXVideo** — LTX-specific nodes
- **ComfyUI-VideoHelperSuite** — VHS_VideoCombine
- **ComfyUI-KJNodes** — VAELoaderKJ, SimpleCalculatorKJ, ImageResizeKJv2, SetNode/GetNode
- **rgthree-comfy** — Power Lora Loader (multi-LoRA support)

## Key Settings

- Sampler: `lcm` (both passes)
- CFG: 1.0
- Pass 1 scheduler: LTXVScheduler (steps=8, max_shift=2.05, min_shift=0.95)
- Pass 2 scheduler: ManualSigmas (0.909375, 0.725, 0.421875, 0.0)
- Camera LoRA: static @ 0.4 strength (default)
- Output: 24fps, H.264, CRF 19

## Requirements

- NVIDIA GPU with 32GB+ VRAM (L40S, A100 80GB, H100 recommended)
- ~60GB disk for models + cache

## Build

```bash
docker build -t edream/gpu-container-ltx:latest --target final .
```

## Local Dev

```bash
docker-compose up  # ComfyUI on :8188, RunPod handler on :8000
```

## Algorithm

`infinidream_algorithm: "ltx-i2v"`
