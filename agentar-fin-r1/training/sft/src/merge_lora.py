"""Merge the trained SFT LoRA adapter back into the base model (ms-swift).

Stage 1 SFT 产出 LoRA adapter（outputs/sft_lora_adapter/），本脚本用 ms-swift 的
`swift export` 把它 merge 回基座，得到完整 checkpoint outputs/sft_merged/，
作为 Stage 2 GRPO 的初始策略（见 grpo/README.md §4.1 方案 A）。

merge 不可逆，adapter 原始副本（sft_lora_adapter/）务必保留。

用法（CLI，推荐）：
    swift export \
        --model Qwen/Qwen3-8B \
        --adapters ./outputs/sft_lora_adapter \
        --torch_dtype float16 \
        --output_dir ./outputs/sft_merged

或 Python API：
    python sft/src/merge_lora.py
"""

from swift.llm import export_main, SwiftExportArguments


def get_export_args() -> SwiftExportArguments:
    return SwiftExportArguments(
        model="Qwen/Qwen3-8B",                       # 基座，FP16 加载
        adapters="./outputs/sft_lora_adapter",        # 训练脚本输出目录
        torch_dtype="float16",
        output_dir="./outputs/sft_merged",            # merge 后的完整 checkpoint
    )


if __name__ == "__main__":
    args = get_export_args()
    export_main(args)
