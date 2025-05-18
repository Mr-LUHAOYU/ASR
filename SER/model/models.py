import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(
            self, input_size=117, num_class=7, hidden_size=512,
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

        # 打包序列以忽略填充部分
        packed_input = pack_padded_sequence(
            t,
            lengths.cpu(),  # 需要将长度放在CPU上
            batch_first=True,
            enforce_sorted=False  # 允许未排序的序列
        )

        # LSTM前向传播
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # 取最后一层的隐藏状态
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # 应用dropout
        last_hidden = self.dropout(last_hidden)

        # 全连接层
        output = self.fc(last_hidden)  # (batch_size, output_dim)

        return output


class MLP(nn.Module):
    def __init__(self, input_dim=6373, num_classes=7, dropout=0.5):
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

        self.fc_out = nn.Linear(512, num_classes)  # 输出7分类

    def forward(self, x, *args, **kwargs):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = self.relu3(x)

        x = self.fc_out(x)
        return x


class CombineModel(nn.Module):
    def __init__(
            self,
            features_dim=6373, temporal_dim=39 * 3,
            num_classes=7, dropout=0.5
    ):
        super(CombineModel, self).__init__()
        self.name = 'combine'
        self.mlp = MLP(features_dim, num_classes, dropout=dropout)
        self.lstm = BiLSTM(temporal_dim, num_classes, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),  # 输入是两路输出的拼接
            # nn.Softmax(dim=1)  # 对两路输出做归一化
        )

    def forward(self, x, t, lengths):
        x_feat = self.mlp(x=x)  # (batch_size, num_classes)
        t_feat = self.lstm(t=t, lengths=lengths)  # (batch_size, temporal_dim)
        feat_concat = torch.cat([x_feat, t_feat], dim=1)  # (batch_size, num_classes + temporal_dim)
        attn_weights = self.attention(feat_concat)  # (batch_size, 1)
        combined = attn_weights * x_feat + (1 - attn_weights) * t_feat  # 动态加权
        return combined
