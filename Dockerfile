FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install uv

# vLLM pinned to 0.10.2 — TRL GRPOTrainer requires exactly this version
RUN uv pip install --system unsloth "vllm==0.10.2" --torch-backend=auto

RUN pip install --no-cache-dir \
    trl \
    transformers \
    accelerate \
    datasets \
    requests \
    peft

WORKDIR /app
COPY train.py .

CMD ["python", "train.py"]
