import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import opensmile
import re


class AudioFeatureExtractor:
    def __init__(self, smile: opensmile.Smile, sr=44100):
        self.sr = sr
        self.smile = smile

    def __call__(self, audio_path):
        return self.extract(audio_path)

    def extract(self, audio_path) -> tuple[pd.DataFrame, str]:
        features = self.smile.process_file(audio_path)
        label = re.match(r"^([A-Za-z]+)\d+\.wav$", os.path.basename(audio_path)).group(1)
        return features, label


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetsPath', type=str, default='../datasets')
    parser.add_argument('--datasetName', type=str, default='SAVEE')
    parser.add_argument('--featuresPath', type=str, default='../features')
    parser.add_argument('--suffix', type=str, default='wav')
    parser.add_argument('--sr', type=int, default=44100)
    args = parser.parse_args()
    return args


def process_and_save(input_root, output_root, suffix, process_func: AudioFeatureExtractor):
    """
    处理输入目录中的文件并保存到输出目录

    :param input_root: 输入根目录 (如 'datasets/data')
    :param output_root: 输出根目录 (如 'processed/data')
    :param suffix: 要处理的文件后缀 (如 '.wav')
    :param process_func: 处理函数，接受输入文件路径，返回要保存的内容
    """
    # 先递归统计所有匹配的文件总数
    total_files = sum(
        len([f for f in files if f.endswith(suffix)])
        for _, _, files in os.walk(input_root)
    )

    # 带进度条的遍历
    with tqdm(total=total_files, desc="Processing files") as pbar:
        speakers = []
        for speaker in os.listdir(input_root):
            speaker_path = os.path.join(input_root, speaker)
            if not os.path.isdir(speaker_path):
                continue

            X: None | pd.DataFrame = None
            y = []
            for filename in os.listdir(speaker_path):
                if not filename.endswith(suffix):
                    continue
                file = os.path.join(speaker_path, filename)
                features, label = process_func(file)
                X = pd.concat([X, features])
                y.append(label)
                pbar.update(1)
            if X is not None and y is not []:
                save_path = os.path.join(output_root, speaker + 'X.csv')
                X.to_csv(save_path, index=False)

                save_path = os.path.join(output_root, speaker + 'y.csv')
                y = pd.DataFrame(y, columns=['label'])
                y.to_csv(save_path, index=False)

                speakers.append(speaker)

        pd.DataFrame(speakers, columns=['speakers']).to_csv(os.path.join(output_root, 'speakers.csv'), index=False)
    print(X.shape)


def extractFromFolder(
        datasetsPath: str, featuresPath: str, datasetName: str,
        extractor: AudioFeatureExtractor, suffix
):
    input_root = os.path.join(datasetsPath, datasetName)
    output_root = os.path.join(featuresPath, datasetName)
    os.makedirs(output_root, exist_ok=True)
    process_and_save(input_root, output_root, suffix, extractor)
    print('all files end with {} are done'.format(suffix))


def main():
    args = get_args()
    print(args)
    datasetsPath = args.datasetsPath
    featuresPath = args.featuresPath
    datasetName = args.datasetName
    sr = args.sr
    suffix = args.suffix
    smile = opensmile.Smile()
    extractor = AudioFeatureExtractor(smile, sr)
    extractFromFolder(
        datasetsPath, featuresPath, datasetName, extractor, suffix
    )


if __name__ == '__main__':
    main()
