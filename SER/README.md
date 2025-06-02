# 语音情感识别系统

## 1. 理论基础

语音情感识别（Speech Emotion Recognition, SER）是通过分析语音信号中的声学特征来识别说话人情感状态的技术，其核心理论可概括为以下四个层次：

### 1.1  声学基础理论
- **情感与声学的关联** 
  不同情感会引发人体发声器官的生理变化，导致声学特征变化：
  
  - **愤怒/快乐**：声带紧张度↑ → 基频（F0）升高、语速加快、能量增大
  - **悲伤**：声带松弛 → 基频降低、语速减慢、能量减弱
  - **恐惧**：呼吸急促 → 高频能量增加、发音抖动（jitter）
  
- **关键声学参数** 
  | 情感类型 | 基频(F0) | 能量 | 语速 | 频谱倾斜 |
  | -------- | -------- | ---- | ---- | -------- |
  | 愤怒     | ↑↑       | ↑↑   | ↑    | 陡峭     |
  | 悲伤     | ↓↓       | ↓    | ↓↓   | 平缓     |

### 1.2. 特征工程
- **传统特征（手工设计）** 
  - **时域特征**：短时能量、过零率
  - **频域特征**：MFCC（梅尔倒谱系数）、F0轮廓、Formant（共振峰）
  - **非线性特征**：HNR（谐噪比）、jitter/shimmer（微扰动）

- **深度特征（自动提取）** 
  通过CNN/Transformer直接从语谱图（Spectrogram）或原始波形中学习高阶表征。


### 1.3. 机器学习方法
- **经典模型** 
  ```mermaid
  graph LR
  A[原始语音] --> B[特征提取]
  B --> C{SVM/GMM/HMM}
  C --> D[情感标签]
  ```

- **深度学习模型** 
  - **CNN**：处理语谱图（如Log-Mel谱）
  - **LSTM**：建模时序动态（如F0轨迹）
  - **端到端模型**（如wav2vec 2.0）直接学习语音-情感映射

### 1.4. 技术挑战
- **跨数据库泛化** 
  不同语料库（如SAVEE vs RAVDESS）的录音条件差异导致模型性能下降。

- **个性化差异** 
  同一情感在不同人语音中表现不同（如男性愤怒基频可能≈女性中性基频）。

- **多模态融合** 
  结合文本语义（ASR转录）或面部表情可提升准确率，但增加系统复杂度。

## 2. 数据集

### 2.1. SAVEE

SAVEE (Surrey Audio-Visual Expressed Emotion) 数据集是一个用于情感识别研究的多模态数据库，主要关注通过语音和面部表情识别人类情感。

SAVEE 数据集共有 7 种情感类别：anger, disgust, fear, happiness, neutral, sadness, surprise

### 2.2. Ravdess

RAVDESS（Ryerson Audio-Visual Database of Emotional Speech and Song）是一个用于情感识别研究的多模态数据库，包含24名专业演员（12名男性，12名女性）表演的语音和歌曲片段.

Ravdess 数据集共有 8 种情感类别：neutral, calm, happy, sad, angry, fearful, disgust, surprised

## 3. 特征提取

### 3.1. 静态特征提取

静态特征使用 `opensmile` 工具包，提取了一组综合特征，得到 6373 维向量。

```python
def extract(self, audio_path) -> tuple[pd.DataFrame, str]:
    features = self.smile.process_file(audio_path)
    label = os.path.basename(audio_path).split('_')[0]
    return features, label
```

### 3.2. 时序特征提取

提取了 MFCC 系数及其一阶、二阶差分，得到 (k, audio_len) 的矩阵。

```python
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
```

### 3.3. 特征存储格式

静态特征保存在 `{speaker}X.csv`

时序特征保存在 `{speaker}T.csv`

语音标签保存在 `{speaker}y.csv`

## 4. 模型构建

### 4.1. RandomForest

使用 `sklearn` 包中的随机森林

```python
def train8val(depth, mfcc, delta, DATASET: str, noise: str):
    dataset = DataSet(f'{DATASET}_{noise}_{mfcc}_{delta}', mfcc=mfcc * delta)
    X_train, y_train, _, _ = dataset.data('train', msg=False)
    X_val, y_val, _, _ = dataset.data('val', msg=False)
    X_test, y_test, _, _ = dataset.data('test', msg=False)

    # 训练模型
    rf = RandomForestClassifier(
        max_depth=depth,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 在验证集上评估
    val_pred = rf.predict(X_val)
    val_score = accuracy_score(y_val, val_pred)

    test_pred = rf.predict(X_test)
    test_score = accuracy_score(y_test, test_pred)
    return val_score, test_score
```

### 4.2. MLP

- 可以处理静态特征输入
- 包含三个隐藏层的感知机模型
- 使用 `ReLu` 作为激活函数
- 带有 `Dropout` 和 `LayerNormal` 提升泛化能力
- 输出为 `n` 个类别

```python
class MLP(nn.Module):
    def __init__(self, input_dim=6373, num_classes=8, dropout=0.5):
        super(MLP, self).__init__()
        self.name = 'mlp'
        self.fc1 = nn.Linear(input_dim, 2048)  # 降维
        self.ln1 = nn.LayerNorm(2048)  # 层归一化
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(2048, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)
        self.relu3 = nn.ReLU()

        self.fc_out = nn.Linear(512, num_classes)
```

### 4.3. LSTM

- 可以处理不定长输入
- 两层双头 LSTM 模型
- 经过一个线性层输出
- 具有 `Dropout` 提高泛化能力
- 输出为 `n` 个类别

```python
class BiLSTM(nn.Module):
    def __init__(
            self, input_size=117, num_class=8, hidden_size=512,
            num_layers=2, dropout=0.5
    ):
        super().__init__()
        self.name = 'lstm'
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_class)
        self.dropout = nn.Dropout(dropout)
    def forward(self, t, lengths, *args, **kwargs):
        """
        :param t: 输入张量 (batch_size, max_seq_len, input_dim)
        :param lengths: 每个序列的实际长度 (batch_size,)
        :return output: 分类结果 (batch_size, output_dim)
        """

        packed_input = pack_padded_sequence(
            t, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_input)
        last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)  # (batch_size, output_dim)
        return output
```

### 4.4. Combine Model

结合上述两种方法，将静态特征输入到MLP模型，将时序特征输入到LSTM模型。

通过一个线性层将二者的输出融合。

```python
class CombineModel(nn.Module):
    def __init__(
            self,
            features_dim=6373, temporal_dim=39 * 3,
            num_classes=8, dropout=0.5
    ):
        super(CombineModel, self).__init__()
        self.name = 'combine'
        self.mlp = MLP(features_dim, num_classes, dropout=dropout)
        self.lstm = BiLSTM(temporal_dim, num_classes, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x, t, lengths):
        x_feat = self.mlp(x=x)
        t_feat = self.lstm(t=t, lengths=lengths)
        feat_concat = torch.cat([x_feat, t_feat], dim=1)
        attn_weights = self.attention(feat_concat)
        combined = attn_weights * x_feat + (1 - attn_weights) * t_feat 
        return combined
```

## 5. 训练方法

所有训练采用 LOSO CV

### 随机加入噪声

```python
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
```

每个视频进行10次随机加噪，扩充数据集。

每次加噪完成一下任务：

1. 随机抽取 0~2 种噪声
2. 对抽取到的噪声使用上述代码加噪
3. 保存音频文件

## 6. 训练过程

### 6.1. SAVEE 训练图
#### 融合模型

![](assets/SAVEE/combine.png)

#### LSTM

![](assets/SAVEE/lstm.png)

### 6.2. Ravdess 训练图

#### 融合模型

![](assets/Ravdess/combine.png)

#### LSTM

![](assets/Ravdess/lstm.png)

### 6.3. 模型测试 (lstm vs combine)

#### SAVEE clean

| delta | mfcc=13 | mfcc=26 | mfcc=39 |
| :--- | :--- | :--- | :--- |
| 1 | 25.00--25.00 | 25.00--25.00 | 25.83--25.00 |
| 2 | 25.00--25.00 | 25.00--25.83 | 25.00--25.00 |
| 3 | 25.00--30.00 | 25.00--25.83 | 25.00--25.00 |

#### SAVEE noise

| delta | mfcc=13 | mfcc=26 | mfcc=39 |
| :--- | :--- | :--- | :--- |
| 1 | 25.83--30.00 | 25.83--30.00 | 25.83--25.00 |
| 2 | 25.83--30.00 | 25.00--25.83 | 25.83--30.00 |
| 3 | 30.00--30.00 | 25.00--25.83 | 25.00--25.00 |

#### Ravdess clean

| delta | mfcc=13      | mfcc=26      | mfcc=39      |
| :---- | :----------- | :----------- | :----------- |
| 1     | 44.17--43.33 | 35.00--41.67 | 47.50--37.50 |
| 2     | 46.67--35.00 | 38.33--49.17 | 52.50--55.00 |
| 3     | 55.00--57.50 | 35.83--49.17 | 53.33--56.67 |

#### Ravdess noise

| delta | mfcc=13      | mfcc=26      | mfcc=39      |
| :---- | :----------- | :----------- | :----------- |
| 1     | 45.83--45.00 | 37.50--45.83 | 33.33--37.50 |
| 2     | 50.00--38.33 | 45.83--52.50 | 52.50--55.00 |
| 3     | 56.67--60.00 | 44.17--53.33 | 55.00--57.50 |

#### 结论

1. 在lstm的基础上融合一个简单的mlp，有助于提高泛化表现
2. 对数据加入噪声、伸缩时间可以提高模型的泛化能力

### 6.4. 模型测试 (RandomForest)

在本实验中，RF 对其参数并不敏感，验证集与测试集的效果在不同参数下相当。

数据集和引入噪声对 RF 的性能有一定作用，实际效果见下表。

| with noise | SAVEE | Ravdess |
| :--------: | :---: | :-----: |
|    Yes     | 25.00 |  50.83  |
|     No     | 23.33 |  54.17  |

说明，SAVEE 数据集比较小，所以分类器的性能要差很多。

## 7. 用户界面

### 7.1. 功能说明

- 支持上传 `wav` 后缀的音频文件
- 支持录音
- 支持播放音频
- 支持裁剪音频
- 支持选择预设的不同模型与参数
- 通过训练好的模型分析输入音频的情感

### 7.2. 情感分析演示

#### 上传音频

![front-1](assets/front-1.png)

#### 参数选择

![front-2](assets/front-2.png)

#### 分析结果

![front-3](assets/front-3.png)

### 7.3. 主要代码

```python
class Page(object):
    def __init__(self, model_zoo, tempfile='temp.wav'):
        self.tempfile = tempfile
        self._upload_init()
        self._model_init()
        self._extractor = Evaluator(model_zoo)

    def _upload_init(self):
        gr.Markdown("## 上传音频文件")
        self.audio = gr.Audio(type="filepath", label="上传WAV文件")

    def _model_init(self):
        gr.Markdown('## 请选择模型')
        self.model = gr.Dropdown(
            ["MLP", "LSTM", "Combine"],
            label="选择模型类型",
            value='Combine',
            interactive=True
        )
        self.dataset = gr.Dropdown(
            ["SAVEE", 'Ravdess'],
            label="选择预训练的数据",
            value='SAVEE',
            interactive=True
        )
        self.noise = gr.Radio(
            ['无噪训练', '带噪训练'],
            label='无噪训练',
            type='index'
        )
        self.mfcc = gr.Dropdown(
            [f'{mfcc * 13}*{delta}'
                for mfcc in range(1, 4)
                for delta in range(1, 4)],
            label='选择MFCC采样参数',
            value=f'39*3',
            interactive=True
        )
        gr.Markdown('## 情感分析结果')
        self.emotion = gr.Markdown('尚未输入语音')
        gr.Button('开始分析').click(
            self.handle,
            inputs=[self.audio, self.model, self.dataset, self.mfcc, self.noise],
            outputs=[self.emotion],
        )

    def handle(self, audio, model, dataset, mfcc, noise):
        print('uploading...')
        self.upload(audio)
        print('done')
        print('calculating emotion')
        noise = 'noise' if noise else 'clean'
        emotion = self.get_emotion(model, dataset, mfcc, noise)
        print('# emotion:', emotion)
        return f'## {emotion}'

    def upload(self, audio):
        shutil.copyfile(audio, self.tempfile)

    def get_emotion(self, model, dataset, mfcc, noise):
        self._extractor.set_model(model, dataset, mfcc, noise)
        emotion = self._extractor(audio=self.tempfile, mfcc=mfcc)
        return str(emotion)
```

## 附录

一些其他脚本和代码

train.sh

```shell
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
```

extract.sh

```shell
mfccs=(13 26 39)
deltas=(1 2 3)
datasetName="Ravdess"
for mfcc in "${mfccs[@]}"; do
  for delta in "${deltas[@]}"; do
    python extract_features.py --datasetName $datasetName \
                              --mfcc $mfcc --delta $delta
  done
done
```

test.py

```python
def test(
        model, dataset: DataSet,
        batch_size: int = 64
):
    model.eval()

    X, y, T, lengths = dataset.data('test', msg=False)
    test_dataset = TensorDataset(X, y, T, lengths)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, labels, t, lengths in test_loader:
            outputs = model(x=x, t=t, lengths=lengths)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = test_correct * 100 / test_total

    return test_acc
```

train8val

```python
def run(
        dataloader, model, criterion, optimizer, device,
        options
):
    if options == 'train':
        model.train()
    elif options == 'val':
        model.eval()

    loss, correct, total = 0.0, 0.0, 0.0
    for x, labels, t, lengths in dataloader:
        x, labels = x.to(device), labels.to(device)
        t, lengths = t.to(device), lengths.to(device)

        outputs = model(x=x, t=t, lengths=lengths)
        loss = criterion(outputs, labels)

        if options == 'train':
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 统计
        loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return loss, correct * 100 / total
```
