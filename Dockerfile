FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Ensure Python output is not buffered
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    python3.10-dev \
    build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu118 \
 && pip install --no-cache-dir \
        transformers==4.53.3 \
        decord==0.6.0 \
        accelerate==1.10.0 \
        bitsandbytes==0.47.0 \
        opencv-python-headless==4.12.0.88 \
        albumentations==2.0.4 \
 && mkdir -p /workspace

# Create a non-root user and set permissions in one layer
# Multiple RUN commands can lead to larger image sizes
RUN groupadd -r user && useradd -m --no-log-init -r -g user user \
    && chown -R user:user /workspace \
    && chmod -R 777 /workspace

# Copy host folders into the image
COPY --chown=user:user ./Models /workspace/Models
COPY --chown=user:user ./task2_runtime /workspace/task2_runtime

# Switch to non-root user
USER user

# Set working directory
WORKDIR /workspace/task2_runtime

# Run inference script by default
ENTRYPOINT ["python", "inference_llava_ov.py"]