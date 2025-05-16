import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from lstm import BiLSTM
from dataset import DataSet
from params import Config

torch.random.manual_seed(42)


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
        probs = torch.softmax(inputs, dim=1)  # 转换为概率 [batch_size, num_classes]

        # 正确类别的概率
        true_probs = 1 - probs[torch.arange(batch_size), labels]  # [batch_size]

        # 错误类别的概率
        mask = torch.ones_like(probs, dtype=bool)
        mask[torch.arange(batch_size), labels] = False
        wrong_probs = probs[mask].view(batch_size, num_classes - 1)  # [batch_size, num_classes-1]

        total_loss = (wrong_probs.pow(2).sum() + true_probs.pow(2).sum()) / batch_size

        return total_loss


def train_model(dataset: DataSet, num_epochs=50, batch_size=16, learning_rate=0.001):
    # 预处理数据
    X, y = dataset.trainData()
    X = torch.tensor(X.to_numpy(), dtype=torch.float)
    y = torch.tensor(Config.encoder.transform(y.squeeze()))
    train_dataset = TensorDataset(X, y)

    X, y = dataset.validData()
    X = torch.tensor(X.to_numpy(), dtype=torch.float)
    y = torch.tensor(Config.encoder.transform(y.squeeze()))
    val_dataset = TensorDataset(X, y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM().to(device)

    # 损失函数和优化器
    criterions = [
        [nn.CrossEntropyLoss(), 1],
        [CustomLoss(), 1],
    ]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)

            loss = 0
            for criterion, weight in criterions:
                loss += criterion(outputs, labels)

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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = 0
                for criterion, weight in criterions:
                    loss += criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        # 打印统计信息
        print('\r', end='')
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%', end='')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Training complete. Best validation accuracy: {best_val_acc:.2f}%')


if __name__ == '__main__':
    data = DataSet('SAVEE')
    data.setValidSpeaker()
    train_model(data)