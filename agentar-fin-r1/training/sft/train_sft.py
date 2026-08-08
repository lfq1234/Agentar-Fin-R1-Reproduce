#!/usr/bin/env python3
# ============================================================================
# Agentar-Fin-R1 — Stage 1 SFT (verl) 训练启动器
# ----------------------------------------------------------------------------
# 本文件持有 Stage 1 SFT 的全部超参，调用 verl 的 SFT 训练入口
# (verl.trainer.sft_trainer)。train_sft.sh 仅负责设置路径环境变量后调用本文件。
#
# 运行：
#   NPROC=8 MODEL_PATH=./Qwen3.5-9B SFT_DATA=./data/verl/sft.parquet \
#     python training/sft/train_sft.py
# ============================================================================
import os
import sys

# ---- 路径（可被环境变量覆盖） ----
MODEL_PATH = os.environ.get("MODEL_PATH", "./Qwen3.5-9B")
SFT_DATA = os.environ.get("SFT_DATA", "./data/verl/sft.parquet")
SAVE_PATH = os.environ.get("SAVE_PATH", "./training/sft/outputs")
NPROC = int(os.environ.get("NPROC", "8"))

# ---- Stage 1 SFT 超参：Qwen3.5-9B + LoRA(r=64, alpha=128, all-linear)，FSDP，8×A800 ----
#   batch = micro_batch(2) × NPROC(8) = 16
#   总步数 = 100K/16 × 2 epochs ≈ 12.5K
#   warmup 400 steps ≈ 3%
SFT_OVERRIDES = [
    f"data.train_files={SFT_DATA}",
    "data.micro_batch_size_per_gpu=2",
    "data.messages_key=messages",
    "data.max_length=4096",
    "optim.lr=1e-4",
    "optim.lr_warmup_steps=400",
    "optim.weight_decay=0.01",
    "optim.lr_scheduler=cosine",
    "engine=fsdp",
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
