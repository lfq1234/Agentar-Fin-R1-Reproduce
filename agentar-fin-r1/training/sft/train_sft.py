#!/usr/bin/env python3
# ============================================================================
# Agentar-Fin-R1 — Stage 1 SFT (verl) 训练启动器
# ----------------------------------------------------------------------------
# 本文件持有 Stage 1 SFT 的全部超参，调用 verl 的 SFT 训练入口
# (verl.trainer.sft_trainer)。train_sft.sh 仅负责设置路径环境变量后调用本文件，
# 形成「shell 设路径 + python 持超参」的清晰分工（与 GRPO 的
# train_grpo.sh + fin_judge_reward.py 对称）。
#
# verl SFT 入口的内部 main 由 @hydra.main 装饰，依赖 sft_trainer_engine.yaml
# 补齐默认配置；因此这里把超参作为 hydra override 传入，既保留 verl 默认结构，
# 又把可调项集中在一处。等价于原 train_sft.sh 的 torchrun 调用。
#
# 运行：
#   NPROC=1 MODEL_PATH=./Qwen3.5-9B SFT_DATA=./data/verl/sft.parquet \
#     python training/sft/train_sft.py
# ============================================================================
import os
import sys

# ---- 路径（可被环境变量覆盖，默认值与 train_sft.sh 一致） ----
MODEL_PATH = os.environ.get("MODEL_PATH", "./Qwen3.5-9B")
SFT_DATA = os.environ.get("SFT_DATA", "./data/verl/sft.parquet")
SAVE_PATH = os.environ.get("SAVE_PATH", "./training/sft/outputs")
NPROC = int(os.environ.get("NPROC", "1"))

# ---- Stage 1 SFT 超参：Qwen3.5-9B + LoRA(r=64, alpha=128, all-linear)，FSDP ----
# 与旧 ms-swift 版映射：
#   tuner_type=lora + lora_rank=64 + lora_alpha=128 + target_modules=all-linear
#     → model.lora_rank / model.lora_alpha / model.target_modules
#   learning_rate=5e-5         → optim.lr
#   num_train_epochs=2         → trainer.total_epochs
#   max_length=8192            → data.max_length
SFT_OVERRIDES = [
    f"data.train_files={SFT_DATA}",
    "data.micro_batch_size_per_gpu=4",
    "data.messages_key=messages",
    "data.max_length=4096",
    "optim.lr=5e-5",
    "optim.lr_warmup_steps=1000",
    "optim.weight_decay=0.01",
    "optim.lr_scheduler=cosine",
    f"model.path={MODEL_PATH}",
    "model.use_remove_padding=true",
    "model.lora_rank=64",
    "model.lora_alpha=128",
    "model.target_modules=all-linear",
    "model.torch_dtype=bfloat16",
    "model.use_gradient_checkpointing=true",
    f"trainer.default_local_dir={SAVE_PATH}",
    "trainer.total_epochs=2",
    'trainer.logger=["console"]',
]


def build_command():
    # torchrun 等价于 python -m torch.distributed.run，保证 torch 安装即可用。
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={NPROC}",
        "-m", "verl.trainer.sft_trainer",
    ] + SFT_OVERRIDES


def main():
    cmd = build_command()
    print("[train_sft.py] launching SFT trainer:", " ".join(cmd), flush=True)
    proc = __import__("subprocess").run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
