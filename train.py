"""
GRPO training on MOA RL environment — gpt-oss 20B BF16 on H100.
Connects to the deployed moa-rl-env on Northflank for rewards.
"""
import os
import requests

from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel
import torch

ENV_URL = os.environ.get("ENV_URL", "https://http--moa-rl-env--7b2fgcxb6gxp.code.run")
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/gpt-oss-20b-instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output/moa-rl-grpo")
TIMEOUT_S = 120

# ── Env helpers ────────────────────────────────────────────────────────────────
def env_reset():
    r = requests.post(f"{ENV_URL}/reset", json={}, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()

def env_step(file_path: str, content: str) -> float:
    r = requests.post(f"{ENV_URL}/step",
                      json={"action": {"file_path": file_path, "content": content}},
                      timeout=TIMEOUT_S)
    r.raise_for_status()
    return float(r.json().get("reward") or 0.0)

# ── Model — BF16, no quantization (H100 has enough VRAM) ──────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=False,
    dtype=torch.bfloat16,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ── Dataset from live env ─────────────────────────────────────────────────────
def build_prompt(obs: dict) -> str:
    return (
        "You are fixing one TypeScript file. Return ONLY the complete fixed file contents.\n\n"
        f"Task:\n{obs['task']}\n\n"
        f"File: {obs['broken_file_path']}\n\n"
        "```ts\n" + obs["broken_file_content"] + "\n```\n\n"
        "Tests:\n```ts\n" + obs["test_file_content"] + "\n```\n"
    )

rows = []
for _ in range(24):
    obs = env_reset()["observation"]
    rows.append({"prompt": build_prompt(obs), "file_path": obs["broken_file_path"]})

dataset = Dataset.from_list(rows)
print(f"Dataset: {len(dataset)} examples")

# ── Reward function ────────────────────────────────────────────────────────────
def _text(completion):
    if isinstance(completion, str): return completion
    if isinstance(completion, list):
        return "".join(c["content"] if isinstance(c, dict) else str(c) for c in completion)
    return str(completion)

def reward_fn(completions, file_path, **kwargs):
    rewards = []
    for completion, fp in zip(completions, file_path):
        code = _text(completion)
        rewards.append(env_step(fp, code))
    return rewards

# ── Training ───────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_fn],
    args=GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_prompt_length=1536,
        max_completion_length=512,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=50,
        max_steps=300,
        bf16=True,
        use_vllm=False,
    ),
    train_dataset=dataset,
)

print(f"Training against {ENV_URL}")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("Done. Model saved to", OUTPUT_DIR)
