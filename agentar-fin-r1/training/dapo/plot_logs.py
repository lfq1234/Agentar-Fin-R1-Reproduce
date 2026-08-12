#!/usr/bin/env python3
"""绘制 DAPO 训练曲线 — 扫描 verl 训练日志（.jsonl）后离线画图。

输出 2×2 综合图：
  - 子图 1: Mean Reward     (critic/score/mean 等)
  - 子图 2: Actor Loss      (actor/loss 等)
  - 子图 3: KL Divergence   (actor/kl 等，注：DAPO 中 KL 进入 reward，kl 值来自 kl_ctrl)
  - 子图 4: Response Length (response_length/mean 等)

风格对齐 training/sft/sft_loss_curve.png（浅色背景、蓝线 + marker + 数值标注 + 网格）。

用法：
  # 默认：扫描 ./training/dapo/outputs/，写到 ./training/dapo/training_curves.png
  python training/dapo/plot_logs.py

  # 自定义目录/输出/标题
  python training/dapo/plot_logs.py --log-dir ./my_logs/ --output ./out.png \
      --title-prefix "DAPO Stage 2"

依赖：matplotlib>=3.8（已加到 training/sft/pyproject.toml 的 observability 可选组）
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ---------- verl 0.8.0 字段候选（不同版本/sub-config 命名有差异，按顺序匹配） ----------
STEP_KEYS = ("step", "training_step", "global_step", "_step")

METRIC_GROUPS = {
    "reward_mean": (
        "critic/score/mean", "reward/mean", "reward/score/mean",
        "val/reward_mean", "reward_mean", "critic/rewards/mean",
    ),
    "actor_loss": (
        "actor/loss", "actor/ppo_loss", "actor/pg_loss", "actor/policy_loss",
    ),
    "kl": (
        "actor/kl", "kl/mean", "actor/kl_loss", "kl_loss", "kl",
    ),
    "response_length": (
        "response_length/mean", "response_length_mean",
        "gen_response_length/mean", "responses/mean_length",
    ),
}
GROUP_TITLES = {
    "reward_mean": "Mean Reward",
    "actor_loss": "Actor Loss",
    "kl": "KL Divergence",
    "response_length": "Mean Response Length",
}


def _find_step_key(seen: set[str]) -> str | None:
    for k in STEP_KEYS:
        if k in seen:
            return k
    return None


def _find_group_key(seen: set[str], candidates: tuple[str, ...]) -> str | None:
    for k in candidates:
        if k in seen:
            return k
    return None


def collect_metrics(log_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """递归扫描 log_dir 下所有 .jsonl，按 step 聚合 metric 序列。

    返回 {metric_name: [(step, value), ...]}（同 step 取最后一次，去重并按 step 升序）。
    """
    series: dict[str, list[tuple[int, float]]] = {}
    jsonl_files = sorted(p for p in log_dir.rglob("*.jsonl") if p.is_file())

    if not jsonl_files:
        print(f"[plot_logs] WARN: {log_dir} 下无 .jsonl 日志", file=sys.stderr)
        return series

    for fp in jsonl_files:
        # 跳过 wandb/tfevents 等大事件流
        if any(part.startswith("wandb") for part in fp.parts):
            continue
        if fp.name.startswith("events.out.tfevents"):
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    seen = set(row.keys())
                    step_key = _find_step_key(seen)
                    if step_key is None:
                        continue
                    try:
                        step = int(row[step_key])
                    except (TypeError, ValueError):
                        continue
                    for k, v in row.items():
                        if k == step_key:
                            continue
                        # 仅收集标量数字（bool 排除）
                        if isinstance(v, bool):
                            continue
                        if isinstance(v, (int, float)):
                            series.setdefault(k, []).append((step, float(v)))
        except OSError as e:
            print(f"[plot_logs] WARN: 读取 {fp} 失败: {e}", file=sys.stderr)

    # 同 step 取最后一次 + 升序
    for k in list(series.keys()):
        d: dict[int, float] = {}
        for s, v in series[k]:
            d[s] = v
        series[k] = sorted(d.items())

    return series


def plot(series: dict, output: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.patch.set_facecolor("#fafafa")

    keys = set(series.keys())
    for idx, group in enumerate(METRIC_GROUPS.keys()):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor("#fafafa")
        metric_key = _find_group_key(keys, METRIC_GROUPS[group])
        if metric_key is None or not series.get(metric_key):
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    fontsize=14, color="gray", transform=ax.transAxes)
            ax.set_title(f"{GROUP_TITLES[group]} (no data)", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        steps = [s for s, _ in series[metric_key]]
        vals = [v for _, v in series[metric_key]]
        ax.plot(steps, vals, color="#1f77b4", linewidth=2.0,
                marker="o", markersize=5, markerfacecolor="#1f77b4")
        for s, v in zip(steps, vals):
            ax.annotate(f"{v:.4f}", xy=(s, v), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=8, color="#333")
        ax.set_title(f"{GROUP_TITLES[group]} ({metric_key})", fontsize=12)
        ax.set_xlabel("Training Step", fontsize=10)
        ax.set_ylabel(group, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{title_prefix} Training Curves (Qwen3-8B + LoRA, DeepFinance-100K)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[plot_logs] 已保存: {output}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines[1])
    p.add_argument("--log-dir", default="./training/dapo/outputs",
                   help="训练日志根目录（递归扫描 .jsonl）")
    p.add_argument("--output", default="./training/dapo/training_curves.png",
                   help="输出图片路径")
    p.add_argument("--title-prefix", default="DAPO Stage 2",
                   help="图表标题前缀")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    print(f"[plot_logs] 扫描: {log_dir.resolve()}")
    series = collect_metrics(log_dir)
    if not series:
        print(f"[plot_logs] 未收集到任何 metrics，请确认训练已运行且 {log_dir} 下有 .jsonl",
              file=sys.stderr)
        sys.exit(1)

    print(f"[plot_logs] 收集到 {len(series)} 个 metric 序列")
    plot(series, Path(args.output), args.title_prefix)


if __name__ == "__main__":
    main()