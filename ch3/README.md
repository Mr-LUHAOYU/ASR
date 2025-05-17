根据实验要求和代码实现，以下是完整的实验报告内容：

# 实验三：语音信号处理实验报告

```python
# 实验环境配置
!pip install soundfile librosa matplotlib pystoi pesq
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
```

## 一、端点检测（双门限法）

### 1. 实验原理

**双门限检测流程**：

```mermaid
graph TD
A[开始] --> B[分帧计算能量和ZCR]
B --> C{能量>ITL 或 ZCR>阈值?}
C -->|是| D[标记语音开始]
C -->|否| E[持续检测]
D --> F{能量<ITU 且 ZCR<阈值?}
F -->|是| G[静音计数+1]
F -->|否| H[重置静音计数]
G --> I{静音帧数≥min_silence?}
I -->|是| J[标记语音结束]
I -->|否| F
```

### 2. 核心实现

```python
def endpoint_detection(
        file_path, frame_len=400, step=160, 
        ITL=0.3, ITU=0.2, ZCR_th=50, min_silence=10
):
    # 读取音频
    y, sr = sf.read(file_path)
    if y.ndim > 1: y = y[:,0]  # 转为单声道
    
    # 预加重
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # 分帧
    frames = []
    n_frames = (len(y) - frame_len) // step + 1
    for i in range(n_frames):
        frames.append(y[i*step : i*step+frame_len])
    
    # 计算短时能量和过零率
    energy = np.array([np.sum(frame**2) for frame in frames])
    zcr = np.array([0.5 * np.sum(np.abs(np.diff(np.sign(frame)))) for frame in frames])
    
    # 归一化
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    
    # 端点检测
    in_speech = False
    speech_segments = []
    silence_count = 0
    start = 0
    
    for i in range(len(energy)):
        if not in_speech:
            if energy[i] > ITL or zcr[i] > ZCR_th:
                start = max(0, i - 2)
                in_speech = True
        else:
            if energy[i] < ITU and zcr[i] < ZCR_th:
                silence_count += 1
                if silence_count >= min_silence:
                    end = i - silence_count
                    speech_segments.append((start*step, end*step))
                    in_speech = False
                    silence_count = 0
            else:
                silence_count = 0
    
    return speech_segments, y, energy, zcr
```


## 二、谱减法降噪

### 1. 算法公式

$$
|\hat{X}(k)| = \max(|Y(k)| - \alpha|\hat{N}(k)|, \beta|\hat{N}(k)|)
$$

其中$\alpha=1.5$为过减因子，$\beta=0.2$为噪声下限

### 2. 关键实现

```python
def spectral_subtraction(noisy, sr, n_fft=512, hop_length=160, win_length=400, noise_frames=5):
    # 计算STFT
    D = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, 
                    win_length=win_length, window='hann')
    mag = np.abs(D)
    
    # 估计噪声谱
    noise_profile = np.mean(mag[:, :noise_frames], axis=1, keepdims=True)
    
    # 谱减法
    mag_enhanced = np.maximum(mag - noise_profile, 0)
    
    # 重建信号
    D_enhanced = mag_enhanced * np.exp(1j * np.angle(D))
    enhanced = librosa.istft(D_enhanced, hop_length=hop_length, 
                            win_length=win_length, window='hann')
    
    return enhanced
```

## 三、维纳滤波降噪

### 1. 算法公式

$$
H(k) = \frac{\xi(k)}{1+\xi(k)}
$$

其中$\xi(k)$为先验信噪比估计

### 2. 关键实现

```python
def wiener_filter(noisy, sr, n_fft=512, hop_length=160, win_length=400, noise_frames=5):
    # 计算STFT
    D = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length,
                    win_length=win_length, window='hann')
    power = np.abs(D)**2
    
    # 估计噪声功率
    noise_power = np.mean(power[:, :noise_frames], axis=1, keepdims=True)
    
    # 维纳滤波
    snr_prior = np.maximum((power - noise_power) / noise_power, 1e-6)
    H = snr_prior / (1 + snr_prior)
    
    # 应用滤波
    D_enhanced = D * H
    enhanced = librosa.istft(D_enhanced, hop_length=hop_length,
                           win_length=win_length, window='hann')
    
    return enhanced
```

