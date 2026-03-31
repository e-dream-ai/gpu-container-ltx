# Stage 1: Base image with ComfyUI + LTX 2.3 dependencies
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libgl1 \
    build-essential \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install comfy-cli
RUN pip install --upgrade pip setuptools wheel
RUN pip install comfy-cli

# Pre-install PyTorch with CUDA 12.4 support
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install \
    --cuda-version 12.4 --nvidia --skip-torch-or-directml

RUN comfy tracking disable

WORKDIR /comfyui

# Install runpod and dependencies
RUN pip install runpod requests websocket-client boto3

# Install ComfyUI-LTXVideo custom nodes
RUN cd custom_nodes && \
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    cd ComfyUI-LTXVideo && \
    pip install -r requirements.txt 2>/dev/null || true

# Install ComfyUI-VideoHelperSuite for video I/O (LoadVideo, SaveVideo, CreateVideo)
RUN cd custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt 2>/dev/null || true

RUN pip cache purge

# Support for network volume
ADD src/extra_model_paths.yaml ./

WORKDIR /

# Add scripts
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy snapshot file
ADD *snapshot*.json /

# Restore snapshot for custom nodes
RUN /restore_snapshot.sh

CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS final

ARG HUGGINGFACE_ACCESS_TOKEN

WORKDIR /comfyui

# Create model directories
RUN mkdir -p models/checkpoints models/text_encoders models/loras \
    models/latent_upscale_models models/vae models/vae_approx

# Download LTX 2.3 checkpoint (FP8 quantized ~22GB)
RUN wget -nv -O models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors \
    https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors

# Download Gemma 3 text encoder (~6GB)
RUN wget -nv -O models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors \
    https://huggingface.co/Comfy-Org/ltx-2/resolve/main/gemma_3_12B_it_fp4_mixed.safetensors

# Download distilled LoRA (for two-stage pipeline)
RUN wget -nv -O models/loras/ltx-2.3-22b-distilled-lora-384.safetensors \
    https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors

# Download spatial upscaler 2x
RUN wget -nv -O models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors

# Download TAESD for LTX 2.3 (tiny VAE for fast ~240px preview during render)
# The "wide" variant produces less blurry previews
RUN wget -nv -O models/vae_approx/taeltx2_3.safetensors \
    https://github.com/madebyollin/taehv/raw/main/safetensors/taeltx2_3.safetensors && \
    wget -nv -O models/vae_approx/taeltx2_3_wide.safetensors \
    https://github.com/madebyollin/taehv/raw/main/safetensors/taeltx2_3_wide.safetensors

# Download camera control LoRAs (from LTX-2 19b — partially compatible with 2.3)
RUN wget -nv -O models/loras/ltx-2-19b-lora-camera-control-dolly-in.safetensors \
    https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors && \
    wget -nv -O models/loras/ltx-2-19b-lora-camera-control-dolly-out.safetensors \
    https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors && \
    wget -nv -O models/loras/ltx-2-19b-lora-camera-control-dolly-left.safetensors \
    https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors && \
    wget -nv -O models/loras/ltx-2-19b-lora-camera-control-dolly-right.safetensors \
    https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors && \
    wget -nv -O models/loras/ltx-2-19b-lora-camera-control-jib-up.safetensors \
    https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors && \
    wget -nv -O models/loras/ltx-2-19b-lora-camera-control-jib-down.safetensors \
    https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors && \
    wget -nv -O models/loras/ltx-2-19b-lora-camera-control-static.safetensors \
    https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors

CMD ["/start.sh"]
