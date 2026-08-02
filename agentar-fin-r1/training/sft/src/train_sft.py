"""Stage 1 SFT training entry for Agentar-Fin-R1 (ms-swift).

两阶段训练的第一阶段：金融知识与能力注入（论文 §3.1 + §3.2 Stage 1）。

框架：ms-swift（CLI / Python API 驱动，不手搓 Dataset / collate_fn / 训练循环）。
- 基座：Qwen3-8B，统一 FP16（torch_dtype=bfloat16 会让 ms-swift 走 bf16；本项目约定
  FP16，故用 torch_dtype='float16'）。
- 方法：LoRA（tuner_type='lora'）。
- 数据：数据集每行一个 messages 数组（{role, content}），assistant 段自带
  <think>...</think> 边界，ms-swift 按模型 chat 模板自动拼装，无需任何手写
  formatting_func / role_map。

用法（推荐 CLI，配置全部在命令行，不写乱七八糟的胶水代码）：
    swift sft \
        --model Qwen/Qwen3-8B \
        --dataset ./train_data.jsonl \
        --tuner_type lora \
        --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 \
        --target_modules all-linear \
        --torch_dtype float16 \
        --learning_rate 1e-4 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --max_length 8192 \
        --gradient_checkpointing true \
        --warmup_steps 50 \
        --logging_steps 10 \
        --save_steps 200 \
        --output_dir ./outputs/sft_lora_adapter \
        --report_to wandb

或 Python API（把同样的参数装进 SwiftSftArguments，调用 sft_main）：
    python sft/src/train_sft.py
"""

from swift.llm import SwiftSftArguments, sft_main


def get_sft_args() -> SwiftSftArguments:
    return SwiftSftArguments(
        model="Qwen/Qwen3-8B",                 # 基座，FP16 加载
        dataset="./train_data.jsonl",          # 本地 jsonl，每行一个 messages 数组
        tuner_type="lora",
        lora_rank=64,                          # 知识注入用 64；轻量指令微调可降到 8
        lora_alpha=128,                        # 经验上 alpha = 2*r
        lora_dropout=0.05,
        target_modules="all-linear",          # 等价于手写 7 个 proj，框架统一管理
        torch_dtype="float16",                 # 统一 FP16
        learning_rate=1e-4,                    # LoRA 典型区间 1e-4 ~ 2e-4
        num_train_epochs=3,
        per_device_train_batch_size=1,         # seq=8K 显存敏感
        gradient_accumulation_steps=16,        # 等效 bs=16
        max_length=8192,
        gradient_checkpointing=True,           # 省显存
        warmup_steps=50,
        logging_steps=10,
        save_steps=200,
        output_dir="./outputs/sft_lora_adapter",
        report_to="wandb",                     # 不需要可视化改 "none"
    )


if __name__ == "__main__":
    args = get_sft_args()
    sft_main(args)
