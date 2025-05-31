from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='SAVEE')
parser.add_argument('--model', type=str, default='combine')
parser.add_argument('--savename', type=str, default=None)
parser.add_argument('--noise', type=str)
parser.add_argument('--mfcc', type=int)
args = parser.parse_args()
dataset = args.dataset
model = args.model
savename = args.savename
if savename is None:
    savename = model

path = Path(f'assets/{dataset.split("_")[0]}/{dataset}')
csv = path / f'{savename}.csv'
png = path / f'{savename}.png'

# 1. 读取数据
df = pd.read_csv(csv)

# 2. 创建画布和双Y轴
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴

# 3. 绘制损失曲线 (左Y轴)
line1 = ax1.plot(df['epoch'], df['train loss'], 'b-', label='Train Loss')
line2 = ax1.plot(df['epoch'], df['val loss'], 'b--', label='Val Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')

# 4. 绘制准确率曲线 (右Y轴)
line3 = ax2.plot(df['epoch'], df['train acc'], 'r-', label='Train Acc')
line4 = ax2.plot(df['epoch'], df['val acc'], 'r--', label='Val Acc')
ax2.set_ylabel('Accuracy (%)', color='r', fontsize=12)
ax2.tick_params(axis='y', labelcolor='r')

# 5. 合并图例
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=4, fontsize=10)

# 6. 添加网格和标题
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_title(f'model : {model}, mfcc : {args.mfcc}, {args.noise}')

# 7. 自动调整布局并保存
plt.tight_layout()
plt.savefig(png, dpi=300)
