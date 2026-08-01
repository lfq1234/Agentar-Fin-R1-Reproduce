"""Merge a trained LoRA adapter back into the FP16 base model.

Stage 1 SFT 产出 LoRA adapter（在 ``output_dir`` 下，含 adapter_model.safetensors
+ adapter_config.json），本脚本把它 merge 回基座，得到完整 checkpoint
``merged_dir/``，作为 Stage 2 GRPO 的初始策略（见 grpo/README.md §4.1 方案 A）。

注意：merge 不可逆，adapter 原始副本务必保留（README §10 风险条 7）。

用法：
    直接改下方“配置区”，然后 `python sft/src/merge_lora.py`
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ===================== 配置区（按需修改）=====================
model_name = "Qwen/Qwen3-8B"              # 基座，FP16 加载（与 train_sft.py 一致）
adapter_dir = "./outputs/sft_lora_adapter"  # 训练脚本输出位置，内含 adapter_config.json
merged_dir = "./outputs/sft_merged"         # merge 后的完整 checkpoint 输出
# ==========================================================


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out = Path(merged_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"输出目录非空：{out}，请先清空或换路径（避免覆盖）")

    logger.info("loading base model from %s (FP16)", model_name)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,                 # 与 SFT 加载一致：FP16
        device_map="auto",
        trust_remote_code=True,
    )

    # 直接指向训练输出目录，PeftModel 自动读取 adapter_config.json
    logger.info("attaching adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(base, adapter_dir)

    logger.info("merging adapter into base weights (不可逆，adapter 副本已保留在 %s)", adapter_dir)
    model = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    logger.info("saving merged model to %s", out)
    model.save_pretrained(out, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(out)
    logger.info("done. sft_merged/ 可直接作为 Stage 2 GRPO 的初始策略。")


if __name__ == "__main__":
    main()
