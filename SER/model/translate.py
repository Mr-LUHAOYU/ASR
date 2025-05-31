import os
import pandas as pd
import re
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--dataset', type=str, default='SAVEE')
parse.add_argument('--model', type=str, default='combine')
parse.add_argument('--savename', type=str, default=None)
args = parse.parse_args()
dataset = args.dataset
model = args.model
savename = args.savename
if savename is None:
    savename = model

# 定义正则表达式匹配模式
pattern = r"Epoch (\d+)/(\d+), Train Loss: ([\d.]+), Train Acc: ([\d.]+)%, Val Loss: ([\d.]+), Val Acc: ([\d.]+)%"
assets = f'./assets/{dataset}/'
if not os.path.exists(assets):
    os.makedirs(assets)

# 读取日志文件
with open('train.log', 'r') as f:
    lines = f.readlines()

result = []

# 逐行处理日志
for line in lines:
    if line.strip() == '':
        continue

    match = re.match(pattern, line)
    if match:
        epoch, total_epochs, train_loss, train_acc, val_loss, val_acc = match.groups()
        # 将数据添加到DataFrame
        result.append(pd.DataFrame([
            {
                'epoch': int(epoch),
                'train loss': float(train_loss),
                'train acc': float(train_acc),
                'val loss': float(val_loss),
                'val acc': float(val_acc)
            }]))

filename = f'{assets}/{savename}.csv'
pd.concat(result, ignore_index=True).to_csv(
    filename, index=False, header=True
)

