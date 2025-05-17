# 示例代码框架
import os
import pandas as pd
import librosa
import numpy as np
from dtw import dtw
from sklearn.metrics import accuracy_score, precision_score, recall_score


# 1. 数据准备
# 加载26个字母的模板语音和测试语音
class Dataset(object):
    def __init__(self, name):
        self.name = name
        self.files = os.listdir(self.name)

    def items(self):
        for file in self.files:
            if file.endswith('.wav'):
                yield file[0], os.path.join(self.name, file)


# 2. 特征提取
def extract_mfcc(wave_file):
    y, sr = librosa.load(wave_file)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T


# 3. DTW距离计算
def calculate_dtw(test_mfcc, template_mfcc):
    dist, _, _, _ = dtw(test_mfcc, template_mfcc, dist=lambda x, y: np.linalg.norm(x - y))
    return dist


# 4. 识别过程
def recognize_letter(tests: Dataset, templates: Dataset):
    result = []

    for letter1, test_file in tests.items():
        test_feat = extract_mfcc(test_file)
        min_dist = float('inf')
        recognized = ''

        for letter2, template_file in templates.items():
            template_feat = extract_mfcc(template_file)
            dist = calculate_dtw(test_feat, template_feat)
            if dist < min_dist:
                min_dist = dist
                recognized = letter2

        result.append([letter1, recognized, test_file])

    return result


def test(noise_type):
    template_data = Dataset('pyttsx3')
    test_data = Dataset(noise_type)
    print('testing dataset: ', noise_type)
    result = recognize_letter(test_data, template_data)
    result = pd.DataFrame(result, columns=['letter1', 'letter2', 'test_file'])
    result.to_csv(noise_type + '.csv', index=False)
    acc = accuracy_score(result['letter1'], result['letter2'])
    print(f'acc={acc}')


def main():
    noise_types = ['white', 'pink', 'impulse', 'harmonic']

    for noise_type in noise_types:
        test(noise_type)


if __name__ == '__main__':
    main()
