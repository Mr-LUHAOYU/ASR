import os
import numpy as np
import librosa
from sklearn.preprocessing import RobustScaler
import argparse
from tqdm import tqdm


class AudioFeatureExtractor:
    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.scaler = RobustScaler()

    def __call__(self, audio_path):
        return self.extract(audio_path)

    def extract(self, audio_path):
        """
        主提取函数
        :param audio_path: 语音文件路径
        :return: (N, D)维特征矩阵，N为时间帧数
        """
        y, sr = librosa.load(audio_path, sr=self.sr)

        mfcc = self._extract_mfcc(y, sr)
        pitch = self._extract_pitch(y, sr)
        volume = self._extract_volume(y, sr)
        timbre = self._extract_timbre(y, sr)

        features = np.concatenate([mfcc, pitch, volume, timbre], axis=0)

        return self._normalize(features)

    def _extract_mfcc(self, y, sr):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        # return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

    def _extract_pitch(self, y, sr):
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=70, fmax=400, fill_na=1
        )
        return f0.reshape(1, -1)

    def _extract_volume(self, y, sr):
        return librosa.feature.rms(y=y)

    def _extract_timbre(self, y, sr):
        S = np.abs(librosa.stft(y))
        timbre_features = np.vstack([
            librosa.feature.spectral_centroid(S=S),
            librosa.feature.spectral_bandwidth(S=S),
            librosa.feature.spectral_rolloff(S=S)
        ])
        return timbre_features

    def _normalize(self, features):
        """特征标准化"""
        if not hasattr(self, 'scaler_mean_'):
            self.scaler.fit(features)
        return self.scaler.transform(features)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetsPath', type=str, required=True)
    parser.add_argument('--datasetName', type=str, required=True)
    parser.add_argument('--featuresPath', type=str, required=True)
    parser.add_argument('--suffix', type=str, default='wav')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--n_mfcc', type=int, default=13)
    args = parser.parse_args()
    return args


def process_and_save(input_root, output_root, suffix, process_func):
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
        for root, _, files in os.walk(input_root):
            for filename in files:
                if filename.endswith(suffix):
                    input_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(root, input_root)
                    output_dir = os.path.join(output_root, relative_path)
                    os.makedirs(output_dir, exist_ok=True)

                    output_path = os.path.join(output_dir, filename).strip(suffix) + 'npz'
                    feature_matrix = process_func(input_path)

                    np.savez_compressed(
                        output_path,
                        features=feature_matrix,
                        timestamp=np.datetime64('now')
                    )
                    pbar.update(1)


def extractFromFolder(
    datasetsPath: str, featuresPath: str, datasetName: str,
    sr, n_mfcc, suffix
):
    extractor = AudioFeatureExtractor(sr=sr, n_mfcc=n_mfcc)
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
    n_mfcc = args.n_mfcc
    suffix = args.suffix
    extractFromFolder(
        datasetsPath, featuresPath, datasetName,
        sr, n_mfcc, suffix
    )


if __name__ == '__main__':
    main()
