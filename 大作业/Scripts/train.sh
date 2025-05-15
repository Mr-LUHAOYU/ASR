#!/bin/bash

# extract features
extract=${extract:-true}
if [[ "${extract,,}" == "true" ]]; then
  cd ../
  python extract_features.py --datasetsPath datasets --featuresPath features \
                           --datasetName SAVEE --suffix wav \
                           --sr 16000 --n_mfcc 13
  cd Scripts/
fi

# train models
pwd
cd ../model/training
python train.py
# python trainer.py --datasetsPath datasets --dataset SAVEE \
#                   --epoch 10 --model-name LSTM \
