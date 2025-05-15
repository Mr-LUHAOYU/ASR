import torch.nn as nn
import torch


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=64, num_layers=4, num_classes=7):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=False)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        # 初始化隐藏状态
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        # 打包序列以忽略填充
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM前向传播
        packed_output, (hn, cn) = self.lstm(packed_input, (h0, c0))

        # 解包序列
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 获取最后一个有效时间步的输出
        last_output = output[torch.arange(batch_size), lengths - 1]

        # 全连接层
        out = self.fc(last_output)

        return out


class BiLSTM(nn.Module):
    def __init__(
        self,
        input_dim=18+13+13,
        hidden_dim=128,
        num_layers=2,
        num_classes=7,
        dropout_prob=0.5,
        bidirectional=True
    ):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM层（双向 + Dropout）
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_prob if num_layers > 1 else 0  # 仅在多层LSTM时生效
        )

        # 全连接层（考虑双向输出的拼接）
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

        # 额外的Dropout层
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, lengths):
        batch_size = x.size(0)

        # 初始化隐藏状态（考虑双向）
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim
        ).to(x.device)

        # 打包序列以忽略填充
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM前向传播
        packed_output, (hn, cn) = self.lstm(packed_input, (h0, c0))

        # 解包序列
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 获取最后一个有效时间步的输出（考虑双向）
        if self.bidirectional:
            # 双向LSTM：拼接前向和后向的最后一个有效输出
            forward_output = output[torch.arange(batch_size), lengths - 1, :self.hidden_dim]
            backward_output = output[torch.arange(batch_size), 0, self.hidden_dim:]
            last_output = torch.cat((forward_output, backward_output), dim=1)
        else:
            # 单向LSTM：直接取最后一个有效输出
            last_output = output[torch.arange(batch_size), lengths - 1]

        # 应用Dropout
        last_output = self.dropout(last_output)

        # 全连接层
        out = self.fc(last_output)

        return out