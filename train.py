"""
GRPO training on MOA RL environment using TRL + Unsloth.
Connects to the deployed moa-rl-env server for rewards.
"""
import asyncio
import os
import httpx
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel

ENV_URL = os.environ.get("ENV_URL", "https://http--moa-rl-env--7b2fgcxb6gxp.code.run")
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Llama-3.1-8B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output/moa-rl-grpo")

# ── Model ──────────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # auto
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ── Tasks dataset ──────────────────────────────────────────────────────────────
def fetch_tasks() -> list[dict]:
    resp = httpx.get(f"{ENV_URL}/tasks", timeout=30)
    resp.raise_for_status()
    return resp.json()["tasks"]

PROMPT_TEMPLATE = """\
You are an expert TypeScript developer.
Fix the following broken file so that all tests pass.

File: {file_path}

Current content:
```typescript
{current_content}
```

Respond with ONLY the fixed TypeScript file contents, no explanation.
"""

def build_dataset() -> Dataset:
    tasks = fetch_tasks()
    rows = []
    for t in tasks:
        prompt = PROMPT_TEMPLATE.format(
            file_path=t["file_path"],
            current_content=t.get("current_content", "// empty"),
        )
        rows.append({"prompt": prompt, "task_id": t["id"], "file_path": t["file_path"]})
    return Dataset.from_list(rows)

dataset = build_dataset()

# ── Reward function ────────────────────────────────────────────────────────────
async def _call_step(session_id: str, file_path: str, content: str) -> float:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{ENV_URL}/step", json={
            "session_id": session_id,
            "action": {"file_path": file_path, "content": content},
        })
        resp.raise_for_status()
        data = resp.json()
        return data["reward"]

async def _reset(task_id: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()["session_id"]

def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    task_ids = kwargs.get("task_id", [None] * len(prompts))
    file_paths = kwargs.get("file_path", [None] * len(prompts))

    async def run_all():
        rewards = []
        for task_id, file_path, completion in zip(task_ids, file_paths, completions):
            try:
                session_id = await _reset(task_id)
                reward = await _call_step(session_id, file_path, completion)
            except Exception as e:
                print(f"[reward_fn] error: {e}")
                reward = 0.0
            rewards.append(reward)
        return rewards

    return asyncio.run(run_all())

# ── Training ───────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[reward_fn],
    args=GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        report_to="none",
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=1024,
    ),
    train_dataset=dataset,
)

print(f"Training on {len(dataset)} tasks against {ENV_URL}")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("Done. Model saved to", OUTPUT_DIR)
