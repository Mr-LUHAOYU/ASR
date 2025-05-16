import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from models import BiLSTM, MLP, CombineModel
from dataset import DataSet
from params import Config
from sklearn.preprocessing import RobustScaler

torch.random.manual_seed(42)


def train_deep_model(dataset: DataSet, num_epochs=1000, batch_size=32, learning_rate=0.001):
    # 预处理数据
    scaler = RobustScaler()

    X, y, T = dataset.trainData()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(Config.encoder.transform(y.squeeze()))
    T = torch.tensor(T.to_numpy(), dtype=torch.float)
    N, D = T.shape
    T = T.reshape(N // 117, 117, D)
    train_dataset = TensorDataset(X, y, T)

    X, y, T = dataset.validData()
    X = scaler.transform(X)
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(Config.encoder.transform(y.squeeze()))
    T = torch.tensor(T.to_numpy(), dtype=torch.float)
    N, D = T.shape
    T = T.reshape(N // 117, 117, D)
    val_dataset = TensorDataset(X, y, T)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = BiLSTM().to(device)
    # model = MLP().to(device)
    model = CombineModel().to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    best_train = [0.0,  0.0]
    best_val = [0.0, 0.0]

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels, temporals in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            temporals = temporals.to(device)

            # 前向传播
            outputs = model(inputs, temporals)

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
            for inputs, labels, temporals in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                temporals = temporals.to(device)

                outputs = model(inputs, temporals)

                loss = criterion(outputs, labels)

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
        if val_acc > best_val[1]:
            best_val = [train_acc, val_acc]
            torch.save(model.state_dict(), 'best_model.pth')

        if train_acc > best_train[0]:
            best_train = [train_acc, val_acc]

    print(f'\nTraining complete.')
    print(f'best on train: train_acc={best_train[0]}, val_acc={best_train[1]}')
    print(f'best on val: train_acc={best_val[0]}, val_acc={best_val[1]}')


if __name__ == '__main__':
    data = DataSet('SAVEE')
    data.setValidSpeaker()
    train_deep_model(data)
    # for i in range(4):
    #     data.setValidSpeaker(i)
    #     train_deep_model(data)

