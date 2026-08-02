"""Stage 2 GRPO training entry for Agentar-Fin-R1 (ms-swift).

在 Stage 1 merge 后的基座（outputs/sft_merged/）上，用 GRPO 做强化学习攻坚
（论文 §3.2 Stage 2）。框架：ms-swift 的 swift rlhf --rlhf_type grpo。

- 基座：FP16 加载（与 SFT 一致）；LoRA 由 tuner_type='lora' 挂载。
- 方法：GRPO（无 Critic）；ref 策略 = 冻结基座，ms-swift 自动处理。
- reward：外部 LLM-as-judge（见 rewards.py，--external_plugins + --reward_funcs）。

推荐 CLI（配置全在命令行，不手搓 rollout / 优势 / loss）：
    swift rlhf \
        --rlhf_type grpo \
        --model ./outputs/sft_merged \
        --dataset ./grpo_data.jsonl \
        --tuner_type lora \
        --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
        --target_modules all-linear \
        --torch_dtype float16 \
        --external_plugins ./grpo/src/rewards.py \
        --reward_funcs judge_reward \
        --num_generations 8 \
        --beta 0.04 \
        --temperature 0.9 --top_p 0.9 \
        --max_completion_length 1024 \
        --max_prompt_length 2048 \
        --learning_rate 5e-6 \
        --lr_scheduler_type constant_with_warmup \
        --warmup_steps 50 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 1 \
        --gradient_checkpointing true \
        --output_dir ./outputs/grpo_lora_adapter \
        --report_to wandb

或 Python API：
    python grpo/src/train_grpo.py
"""

from swift.llm import RLHFArguments, rlhf_main

# 裁判模型服务（OpenAI 兼容 /v1，独立部署，不与训练抢显存）。
JUDGE_BASE_URL = "http://localhost:8000/v1"
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
JUDGE_API_KEY = "EMPTY"


def get_grpo_args() -> RLHFArguments:
    return RLHFArguments(
        rlhf_type="grpo",
        model="./outputs/sft_merged",           # 初始策略 = Stage 1 merge 后的完整模型
        dataset="./grpo_data.jsonl",            # 每行含 messages + gold_answer 列
        tuner_type="lora",
        lora_rank=32,                           # 比 SFT 的 64 小，RL 阶段防过大更新
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        torch_dtype="float16",                  # 与 SFT 一致：FP16
        # —— GRPO 专属 ——
        reward_funcs="judge_reward",            # 注册在 rewards.py 的 orms 键
        external_plugins="./grpo/src/rewards.py",
        num_generations=8,                      # group size K：每 prompt 采 8 条 rollout
        beta=0.04,                              # KL 系数：约束策略别漂离 ref 太远
        temperature=0.9,                        # rollout 探索温度
        top_p=0.9,
        max_completion_length=1024,             # CoT 需要空间
        max_prompt_length=2048,
        # —— 优化 ——
        learning_rate=5e-6,                     # RL 阶段 lr 远小于 SFT，防策略崩溃
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        per_device_train_batch_size=1,          # 单 prompt
        gradient_accumulation_steps=8,          # 等效 8 prompts/step
        num_train_epochs=1,                     # RL 易过拟合 reward，少 epoch
        gradient_checkpointing=True,            # rollout 阶段省显存
        output_dir="./outputs/grpo_lora_adapter",
        report_to="wandb",                      # 不需要可视化改 "none"
    )


if __name__ == "__main__":
    args = get_grpo_args()
    rlhf_main(args)
