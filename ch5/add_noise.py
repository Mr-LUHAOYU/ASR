import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import librosa

# 设置随机种子保证可重复性
np.random.seed(42)


def load_audio(file_path):
    """加载WAV文件"""
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


def save_audio(audio, sr, file_path):
    """保存为WAV文件"""
    sf.write(file_path, audio, sr)


def add_white_noise(audio, noise_level=0.01):
    """添加白噪声"""
    noise = np.random.normal(0, noise_level * np.max(audio), len(audio))
    return audio + noise


def add_pink_noise(audio, noise_level=0.01):
    """添加粉红噪声（1/f噪声）"""
    uneven = noise_level * np.random.randn(len(audio))
    f = np.fft.rfftfreq(len(audio))
    f[0] = 0.001  # 避免除以零
    pink_filter = 1 / np.sqrt(f)
    pink_filter = pink_filter / np.max(pink_filter)
    fft = np.fft.rfft(uneven)
    pink_noise = np.fft.irfft(fft * pink_filter, n=len(audio))
    return audio + pink_noise * np.max(audio) * 0.1


def add_impulse_noise(audio, probability=0.001, strength=0.5):
    """添加脉冲噪声（咔嗒声）"""
    noisy_audio = audio.copy()
    for i in range(len(noisy_audio)):
        if random.random() < probability:
            noisy_audio[i] += strength * (2 * random.random() - 1) * np.max(audio)
    return noisy_audio


def add_harmonic_distortion(audio, distortion_level=0.1):
    """添加谐波失真"""
    return audio + distortion_level * np.sin(2 * np.pi * 1000 * np.arange(len(audio))) / len(audio) * audio


def process_letter_files(input_dir, noise_types=None):
    """
    处理字母发音文件，添加噪声
    :param input_dir: 包含A.wav-Z.wav的输入目录
    :param noise_types: 要添加的噪声类型列表，可选['white', 'pink', 'impulse', 'harmonic']
    """
    if noise_types is None:
        noise_types = ['white', 'pink', 'impulse', 'harmonic']

    for noise_type in noise_types:
        os.makedirs(noise_type, exist_ok=True)

    # 获取所有字母文件
    letter_files = [f for f in os.listdir(input_dir) if f.endswith('.wav') and f[0].isalpha()]

    for letter_file in tqdm(letter_files, desc="Processing letters"):
        letter = os.path.splitext(letter_file)[0].upper()
        input_path = os.path.join(input_dir, letter_file)

        # 加载原始音频
        audio, sr = load_audio(input_path)

        # 为每种噪声类型创建加噪版本
        for noise_type in noise_types:
            noisy_audio = audio.copy()
            for i in range(5):
                if noise_type == 'white':
                    noise_level = random.uniform(0.01, 0.05)
                    noisy_audio = add_white_noise(
                        noisy_audio,
                        noise_level=noise_level
                    )
                elif noise_type == 'pink':
                    noise_level = random.uniform(0.01, 0.05)
                    noisy_audio = add_pink_noise(
                        noisy_audio,
                        noise_level=noise_level
                    )
                elif noise_type == 'impulse':
                    probability = random.uniform(0.001, 0.005)
                    strength = random.uniform(0.3, 0.7)
                    noisy_audio = add_impulse_noise(
                        noisy_audio,
                        probability=probability,
                        strength=strength
                    )
                elif noise_type == 'harmonic':
                    distortion_level = random.uniform(0.05, 0.15)
                    noisy_audio = add_harmonic_distortion(
                        noisy_audio,
                        distortion_level=distortion_level
                    )
                else:
                    continue

                # 确保音频在-1到1范围内
                noisy_audio = np.clip(noisy_audio, -0.99, 0.99)

                # 保存加噪音频
                filename = f'{letter}_{i}.wav'
                output_path = os.path.join(noise_type, filename)

                save_audio(noisy_audio, sr, output_path)


if __name__ == "__main__":
    # 配置路径
    input_directory = "pyttsx3"  # 存放原始A.wav-Z.wav的目录

    # 运行处理
    process_letter_files(
        input_directory,
    )

    print('处理完成！')