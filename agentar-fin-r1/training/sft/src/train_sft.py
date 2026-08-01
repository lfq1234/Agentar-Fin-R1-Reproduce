"""Stage 1 SFT training entry for Agentar-Fin-R1.

两阶段训练的第一阶段：金融知识与能力注入（论文 §3.1 + §3.2 Stage 1）。
- 基座：FP16 加载 Qwen3-8B（transformers.AutoModelForCausalLM）
- 方法：PEFT + LoRA
- 数据：TRL SFTTrainer + datasets.load_dataset + formatting_func
       （数据已是 messages 数组格式，assistant 段含 <think> 边界，原样保留）

直接复用 TRL 的 SFTTrainer，不自己写 Dataset / collate_fn / 训练循环。
修改下方“配置区”即可切换模型 / 数据 / 超参，然后 `python sft/src/train_sft.py`。
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ===================== 配置区（按需修改）=====================
model_name = "Qwen/Qwen3-8B"            # 基座，FP16 加载
dataset_path = "./train_data.jsonl"     # 本地 json/jsonl，每行一个 messages 数组
output_dir = "./outputs/sft_lora_adapter"

lora_config = LoraConfig(
    r=64,                              # LoRA 秩；轻量指令微调可降到 8，知识注入用 64
    lora_alpha=128,                    # 经验上 alpha = 2*r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,     # seq=8K 显存敏感
    gradient_accumulation_steps=16,    # 等效 bs=16
    learning_rate=1e-4,                # LoRA 典型区间 1e-4 ~ 2e-4
    num_train_epochs=2,                # 100k 干净 CoT：LoRA 过拟合慢，2~3 轮足够；再多易背记/遗忘
    logging_steps=10,
    save_strategy="epoch",             # 2 轮 → 每轮末存一次，共 2 个 checkpoint，不多
    fp16=True,                         # 精度约定：FP16 + AMP
    bf16=False,                        # 不用 bf16
    optim="adamw_torch",
    report_to="wandb",                 # 不需要可视化可改 "none"，并去掉 import wandb
    run_name="qwen3-sft-lora",
    gradient_checkpointing=True,      # 省显存
    warmup_steps=50,
)
# 数据角色名映射（HUMAN/ASSISTANT → Qwen chat 标准角色）
role_map = {"HUMAN": "user", "USER": "user",
            "ASSISTANT": "assistant", "MODEL": "assistant"}
# ==========================================================


def format_messages(sample):
    """把 [{role, content}] 拼成 Qwen chat 文本。

    assistant 段已含 <think>...</think> 边界，原样保留、不重复包裹，
    只按 Qwen 模板补 <|im_start|>/<|im_end|> 边界。
    """
    msgs = sample["messages"] if isinstance(sample, dict) else sample
    parts = []
    for m in msgs:
        role = role_map.get(m["role"].upper(), m["role"].lower())
        parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>\n")
    return "".join(parts)


def main():
    # 加载数据集
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"数据集总量: {len(dataset)}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 加载 FP16 基座 + 挂载 LoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,       # 统一 FP16 加载
        device_map="auto",               # 多卡 DDP 训练时去掉本行，交给 Trainer
        trust_remote_code=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # SFT 训练器（TRL 原生，formatting_func 负责拼 prompt）
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=format_messages,
        max_seq_length=8192,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 开始训练
    trainer.train()
    # 保存 LoRA 权重（adapter 独立保存，不污染基座）
    trainer.save_model(output_dir)
    print(f"LoRA 权重保存至: {output_dir}")


if __name__ == "__main__":
    main()
