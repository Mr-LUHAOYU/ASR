import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import opensmile
import re
import librosa


class AudioFeatureExtractor(object):
    def __init__(self, smile: opensmile.Smile):
        self.smile = smile

    def __call__(self, audio_path):
        return self.extract(audio_path)

    def extract(self, audio_path) -> tuple[pd.DataFrame, str]:
        features = self.smile.process_file(audio_path)
        # label = re.match(r"^([A-Za-z]+)_\d+\.wav$", os.path.basename(audio_path)).group(1)
        label = os.path.basename(audio_path).split('_')[0]
        return features, label


class AudioTemporalExtractor(object):
    def __init__(self, sr=44100, n_mfcc=39):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def __call__(self, audio_path, delta=1):
        return self.extract(audio_path, delta=delta)
        # return self._extractAudio(audio_path)

    def extract(self, audio_path, delta=1) -> pd.DataFrame:
        y, sr = librosa.load(audio_path, sr=self.sr)
        if delta == 1:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features = np.array(mfcc)
        if delta == 2:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            delta_mfcc = librosa.feature.delta(mfcc)
            features = np.vstack([mfcc, delta_mfcc])
        if delta == 3:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

        return pd.DataFrame(features)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetsPath', type=str, default='../datasets')
    parser.add_argument('--datasetName', type=str, default='SAVEE')
    parser.add_argument('--featuresPath', type=str, default='../features')
    parser.add_argument('--suffix', type=str, default='wav')
    parser.add_argument('--sr', type=int, default=48000)
    parser.add_argument('--mfcc', type=int, default=13)
    parser.add_argument('--delta', type=int, default=3)
    args = parser.parse_args()
    return args


def process_and_save(
        input_root, output_root, suffix,
        featureExtractor: AudioFeatureExtractor,
        temporalExtractor: AudioTemporalExtractor,
        delta: int
):
    """
    处理输入目录中的文件并保存到输出目录

    :param input_root: 输入根目录 (如 'datasets/data')
    :param output_root: 输出根目录 (如 'features/data')
    :param suffix: 要处理的文件后缀 (如 '.wav')
    :param featureExtractor: 提取静态特征
    :param temporalExtractor: 提取时序特征
    :param delta: mfcc阶数
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
            temporals = []
            for filename in os.listdir(speaker_path):
                if not filename.endswith(suffix):
                    continue

                file = os.path.join(speaker_path, filename)
                features, label = featureExtractor(file)
                temporal = temporalExtractor(file, delta)

                X = pd.concat([X, features])
                y.append(label)
                temporals.append(temporal)

                pbar.update(1)

            if X is not None and y is not []:
                save_path = os.path.join(output_root, speaker + 'X.csv')
                X.to_csv(save_path, index=False)

                save_path = os.path.join(output_root, speaker + 'y.csv')
                y = pd.DataFrame(y, columns=['label'])
                y.to_csv(save_path, index=False)

                save_path = os.path.join(output_root, speaker + 'T.csv')
                temporals = pd.concat(temporals)
                temporals.to_csv(save_path, index=False)

                speakers.append(speaker)

        pd.DataFrame(speakers, columns=['speakers']).to_csv(os.path.join(output_root, 'speakers.csv'), index=False)

    print(X.shape)


def extractFromFolder(
        datasetsPath: str, featuresPath: str, datasetName: str,
        featureExtractor, temporalExtractor, suffix, mfcc, delta
):
    for p in ['train', 'val', 'test']:
        input_root = os.path.join(datasetsPath, datasetName, p)
        output_root = os.path.join(featuresPath, f'{datasetName}_clean_{mfcc}_{delta}', p)
        os.makedirs(output_root, exist_ok=True)
        process_and_save(
            input_root, output_root, suffix,
            featureExtractor, temporalExtractor,
            delta
        )
    print('all files end with {} are done'.format(suffix))


def main():
    args = get_args()
    print(args)
    datasetsPath = args.datasetsPath
    featuresPath = args.featuresPath
    datasetName = args.datasetName
    sr = args.sr
    suffix = args.suffix
    mfcc = args.mfcc
    delta = args.delta
    smile = opensmile.Smile()
    featureExtractor = AudioFeatureExtractor(smile)
    temporalExtractor = AudioTemporalExtractor(sr, n_mfcc=mfcc)
    extractFromFolder(
        datasetsPath, featuresPath, datasetName,
        featureExtractor, temporalExtractor, suffix,
        mfcc, delta
    )


if __name__ == '__main__':
    main()
