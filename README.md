# gpu-container-ltx

RunPod serverless container for LTX 2.3 video generation via ComfyUI.

## Models Included

| Model | Size | Purpose |
|-------|------|---------|
| `ltx-2.3-22b-dev-fp8.safetensors` | ~22GB | Main checkpoint (FP8 quantized) |
| `gemma_3_12B_it_fp4_mixed.safetensors` | ~6GB | Text encoder |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | ~1GB | Distilled LoRA for two-stage pipeline |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ~1GB | 2x latent upscaler |
| 7x camera control LoRAs | ~100MB each | Dolly in/out/left/right, jib up/down, static |

## Requirements

- NVIDIA GPU with 32GB+ VRAM (L40S, A100 80GB, H100 recommended)
- ~100GB disk for models + cache

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

## Camera Control LoRAs

These are from LTX-2 (19b) and are partially compatible with LTX 2.3 (22b):
- Dolly in/out: confirmed working
- Dolly left/right: may be unreliable
- Jib up/down: untested on 2.3
- Static: confirmed working (Jef recommends even at 20% strength)

## Status

**First cut** — waiting on Jef for optimized workflows and LoRA recommendations.
The test_input.json contains a simplified I2V workflow. The official two-stage workflow
with latent upscaling will be added once tested.
