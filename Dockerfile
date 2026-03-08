FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    trl \
    httpx \
    datasets \
    transformers \
    accelerate \
    peft \
    bitsandbytes

WORKDIR /app
COPY train.py .

CMD ["python", "train.py"]
