"""Merge the Stage-1 SFT LoRA adapter into the base model (verl path).

verl SFT（verl.trainer.sft_trainer + model.lora_rank）产出的 adapter 目录为
peft 兼容格式（adapter_config.json + adapter_model.safetensors）。本脚本用 peft
把它 merge 回基座，得到完整 checkpoint（outputs/sft_merged），作为 Stage 2 GRPO
的初始策略——等价于旧 ms-swift 版 training/sft/src/merge_lora.py（swift export）。

前提：verl SFT 输出是标准 peft adapter。若 verl 版本把 LoRA 存成非 peft 格式，
改用「SFT 阶段也走全参（去掉 model.lora_rank），GRPO 直接以 SFT checkpoint
为基座 + 自己的 LoRA」这一路径，跳过 merge。

用法：
    python training/verl/merge_lora.py \
        --base Qwen/Qwen3-8B \
        --adapter ./outputs/sft_lora_adapter \
        --output ./outputs/sft_merged
"""

import argparse

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description="Merge SFT LoRA adapter into base model")
    p.add_argument("--base", default="Qwen/Qwen3-8B", help="基座模型（与 SFT 一致）")
    p.add_argument("--adapter", default="./outputs/sft_lora_adapter", help="verl SFT 输出的 LoRA 目录")
    p.add_argument("--output", default="./outputs/sft_merged", help="merge 后完整 checkpoint 目录")
    args = p.parse_args()

    print(f"[merge] loading adapter from {args.adapter}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter, torch_dtype="auto", low_cpu_mem_usage=True
    )
    print("[merge] merging LoRA into base weights ...")
    model = model.merge_and_unload()
    model.save_pretrained(args.output)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.save_pretrained(args.output)
    print(f"[merge] done → {args.output}")


if __name__ == "__main__":
    main()
