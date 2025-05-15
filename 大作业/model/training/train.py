import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import torch
import torch.nn as nn
sys.path.append('f:/ASR/大作业/model')
from models.lstm import LSTMClassifier, BiLSTM
from data.dataset import DataSet


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, labels):
        """
        Args:
            inputs: Tensor of shape [batch_size, num_classes] (logits or probabilities)
            labels: Tensor of shape [batch_size] (ground truth class indices)
        Returns:
            loss: Scalar tensor
        """
        batch_size, num_classes = inputs.shape
        probs = inputs
        # probs = torch.softmax(inputs, dim=1)  # 转换为概率 [batch_size, num_classes]

        # 正确类别的概率
        true_probs = probs[torch.arange(batch_size), labels]  # [batch_size]

        # 错误类别的概率
        mask = torch.ones_like(probs, dtype=bool)
        mask[torch.arange(batch_size), labels] = False
        wrong_probs = probs[mask].view(batch_size, num_classes - 1)  # [batch_size, num_classes-1]

        # 损失计算
        # loss_true = (1 - true_probs).pow(2).mean()  # 鼓励正确类别概率接近 1
        # loss_wrong = wrong_probs.pow(2).mean()  # 鼓励错误类别概率接近 0

        total_loss = (wrong_probs.sum() - true_probs.sum() * 2) / batch_size
        # print(total_loss)
        # exit(0)
        return total_loss


def train_model(dataset: DataSet, num_epochs=50, batch_size=16, learning_rate=0.001):
    # 预处理数据
    X, y, lengths = dataset.trainData()
    train_dataset = TensorDataset(X, y, lengths)
    X, y, lengths = dataset.validData()
    val_dataset = TensorDataset(X, y, lengths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = LSTMClassifier().to(device)
    model = BiLSTM().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels, lengths in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels, lengths in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        # 打印统计信息
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Training complete. Best validation accuracy: {best_val_acc:.2f}%')


if __name__ == '__main__':
    data = DataSet('SAVEE')
    data.setValidSpeaker()
    train_model(data)