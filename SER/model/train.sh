#!/bin/bash

# 参数设置
epoch=100
batchSize=64
mfccs=(13 26 39)
noise=clean
deltas=(1 2 3)
models=("combine" "lstm" "mlp")

for mfcc in "${mfccs[@]}"; do
  for delta in "${deltas[@]}"; do
    dataset="SAVEE_${noise}_${mfcc}_${delta}"

    for model in "${models[@]}"; do
      echo "$dataset $model"

      # 训练模型
      python train.py --dataset "$dataset" --model "$model" \
                     --epoch "$epoch" --batchSize "$batchSize" \
                     --mfcc $((mfcc*delta))

      # 转写
      python translate.py --dataset "$dataset" --model "$model"

      # 绘图
      python plot.py --dataset "$dataset" --model "$model" \
                    --mfcc $((mfcc*delta)) --noise $noise
    done
  done
done