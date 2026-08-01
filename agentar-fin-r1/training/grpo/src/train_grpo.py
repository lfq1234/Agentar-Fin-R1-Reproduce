"""Stage 2 GRPO training entry for Agentar-Fin-R1.

在 Stage 1 merge 后的基座（outputs/sft_merged/）上，用 GRPOTrainer + LoRA
做强化学习攻坚（论文 §3.2 Stage 2）。

- 基座：FP16 加载（与 SFT 一致）；LoRA 由 GRPOTrainer 通过 peft_config 挂载
- 方法：GRPO（无 Critic）；ref 策略 = 冻结基座，TRL 在 peft_config 存在时自动处理
- reward：见 rewards.py（按 answer_type 分派 + 格式奖励）

直接复用 TRL 的 GRPOTrainer，不自己写 rollout / 优势 / loss。
修改下方“配置区”，然后 `python grpo/src/train_grpo.py`。
"""
from __future__ import annotations

import json

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from rewards import LLMJudge, make_reward_funcs

# ===================== 配置区（按需修改）=====================
model_name = "./outputs/sft_merged"    # 初始策略 = Stage 1 merge 后的完整模型（FP16 加载）
dataset_path = "./grpo_data.jsonl"     # 每行: query / gold_answer
output_dir = "./outputs/grpo_lora_adapter"

# 裁判模型服务（OpenAI 兼容 /v1，推荐 vLLM 独立部署，不与训练抢显存）
judge_base_url = "http://localhost:8000/v1"
judge_model = "Qwen/Qwen2.5-72B-Instruct"   # 服务实际加载的裁判模型
judge_api_key = "EMPTY"                      # vLLM 默认；接商业 API 时填对应 key

lora_config = LoraConfig(
    r=32,                              # 比 SFT 的 64 小，RL 阶段防过大更新
    lora_alpha=64,                     # alpha = 2*r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

grpo_config = GRPOConfig(
    output_dir=output_dir,
    learning_rate=5e-6,                # RL 阶段 lr 远小于 SFT，防策略崩溃
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    num_generations=8,                 # group size K：每 prompt 采 8 条 rollout 互比优势
    beta=0.04,                         # KL 系数：约束策略别漂离 ref 太远（防 reward hacking）
    max_prompt_length=2048,
    max_completion_length=1024,        # CoT 需要空间
    temperature=0.9,                   # rollout 探索温度；后期可退火到 0.5
    top_p=0.9,
    per_device_train_batch_size=1,     # 单 prompt
    gradient_accumulation_steps=8,     # 等效 8 prompts/step
    num_train_epochs=1,                # RL 易过拟合 reward，少 epoch（1~2）
    fp16=True,                         # 与 SFT 一致：FP16 + AMP
    bf16=False,
    gradient_checkpointing=True,       # rollout 阶段省显存
    report_to="wandb",                 # 不需要可视化改 "none"，并去掉 import wandb
    run_name="qwen3-grpo-lora",
)
# ==========================================================


def build_dataset(path):
    """读 jsonl，把 query 转成 GRPO 需要的 prompt（messages 列表）。

    同时保留 query / gold_answer 列，随 batch 透传给 rewards.py 的裁判 reward。
    数据集每行: {"query": ..., "gold_answer": ...}
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            rows.append({
                "prompt": [{"role": "user", "content": ex["query"]}],
                "query": ex["query"],
                "gold_answer": ex.get("gold_answer", ""),
            })
    return Dataset.from_list(rows)


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 裁判模型服务（独立部署，HTTP 调用，不与训练抢显存）
    judge = LLMJudge(base_url=judge_base_url, model=judge_model, api_key=judge_api_key)

    # FP16 加载 Stage 1 merge 后的基座；LoRA 由 GRPOTrainer 通过 peft_config 挂载
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,       # 与 SFT 加载一致：FP16
        device_map="auto",               # 多卡 DDP 训练时去掉本行，交给 Trainer
        trust_remote_code=True,
    )

    dataset = build_dataset(dataset_path)
    print(f"GRPO 数据集总量: {len(dataset)}")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=make_reward_funcs(judge),   # LLM 裁判打分
        peft_config=lora_config,                 # 挂 LoRA；ref = 冻结基座（TRL 自动）
    )

    trainer.train()
    # 保存 LoRA 权重（adapter 独立保存，不污染基座）
    trainer.save_model(output_dir)
    print(f"GRPO LoRA 权重保存至: {output_dir}")


if __name__ == "__main__":
    main()
