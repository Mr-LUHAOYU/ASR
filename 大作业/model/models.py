import torch
import torch.nn as nn


def calLen(X):
    ...


class BiLSTM(nn.Module):
    def __init__(self, input_size=117, num_class=7, hidden_size=512, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # x是列表[tensor(seq_len1, 117), tensor(seq_len2, 117), ...]

        # 1. 拼接所有序列并记录长度
        lengths = [calLen(seq) for seq in x]
        # print(lengths)
        # exit(0)
        padded_x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)  # (batch, max_len, 117)

        # 2. 使用pack_padded_sequence处理变长序列
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            padded_x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # 3. LSTM处理
        packed_out, (h_n, _) = self.lstm(packed_x)  # h_n形状: (num_layers, batch, hidden_dim)

        # 4. 取最后一层的隐藏状态（已经是每个序列的最后时间步）
        last_hidden = h_n[-1]  # (batch, hidden_dim)

        # 5. 全连接层
        return self.fc(last_hidden)  # (batch, 7)


class MLP(nn.Module):
    def __init__(self, input_dim=6373, num_classes=7):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)  # 降维
        self.ln1 = nn.LayerNorm(2048)  # 层归一化
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(2048, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)
        self.relu3 = nn.ReLU()

        self.fc_out = nn.Linear(512, num_classes)  # 输出7分类

    def forward(self, x):
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
    def __init__(self, features_dim=6373, temporal_dim=39 * 3, num_classes=7):
        super(CombineModel, self).__init__()
        self.mlp = MLP(features_dim, num_classes)
        self.lstm = BiLSTM(features_dim, temporal_dim)
        self.attention = nn.Sequential(
            nn.Linear(num_classes + temporal_dim, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, X, T):
        x_feat = self.mlp(X)  # (batch_size, num_classes)
        t_feat = self.lstm(T)  # (batch_size, temporal_dim)
        feat_concat = torch.cat([x_feat, t_feat], dim=1)  # (batch_size, num_classes + temporal_dim)
        attn_weights = self.attention(feat_concat)  # (batch_size, 1)
        combined = attn_weights * x_feat + (1 - attn_weights) * t_feat  # 动态加权
        return self.fc(combined)
