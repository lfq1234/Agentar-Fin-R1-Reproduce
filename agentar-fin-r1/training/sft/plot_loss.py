"""绘制 SFT loss 曲线"""

import matplotlib.pyplot as plt

steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6190]
loss = [0.6692, 0.6257, 0.6157, 0.6040, 0.6000, 0.5963, 0.5325, 0.5116, 0.5100, 0.5028, 0.5056, 0.5043, 0.5014]

plt.figure(figsize=(10, 5))
plt.plot(steps, loss, marker='o', linewidth=1.5, markersize=6, color='#1f77b4')
plt.title('SFT Training Loss Curve (Qwen3-8B + LoRA, 100K DeepFinance-100K)', fontsize=13)
plt.xlabel('Training Step')
plt.ylabel('Average Loss (per 500 steps)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

for x, y in zip(steps, loss):
    plt.annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(0, 8),
                 ha='center', fontsize=8, color='#333')

plt.savefig('sft_loss_curve.png', dpi=150, bbox_inches='tight')
print('Done')