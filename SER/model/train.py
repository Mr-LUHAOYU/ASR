import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from models import BiLSTM, MLP, CombineModel
from dataset import DataSet
from params import config

torch.random.manual_seed(42)


def logPrint(*args, **kwargs):
    if kwargs.get('clear', False):
        file = open(config.trainLog, 'w')
        print(file=file)
        file.close()
    else:
        file = open(config.trainLog, 'a')
        print(*args, **kwargs)
        print(*args, **kwargs, file=file)
        file.close()


def train_deep_model(
        dataset: DataSet, model: str = 'combine',
        num_epochs=10, batch_size=32, learning_rate=0.001,
        dropout=0.5
):
    # 预处理数据
    X, y, T, lengths = dataset.trainData()
    train_dataset = TensorDataset(X, y, T, lengths)

    X, y, T, lengths = dataset.validData()
    val_dataset = TensorDataset(X, y, T, lengths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model.lower() == 'combine':
        model = CombineModel(dropout=dropout).to(device)
    elif model.lower() == 'bilstm':
        model = BiLSTM(dropout=dropout).to(device)
    elif model.lower() == 'mlp':
        model = MLP(dropout=dropout).to(device)
    else:
        raise Exception('Unknown model')

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    best_train = [0.0, 0.0]
    best_val = [0.0, 0.0]

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, labels, t, lengths in train_loader:
            x, labels = x.to(device), labels.to(device)
            t, lengths = t.to(device), lengths.to(device)

            outputs = model(x=x, t=t, lengths=lengths)

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
            for x, labels, t, lengths in val_loader:
                x, labels = x.to(device), labels.to(device)
                t, lengths = t.to(device), lengths.to(device)

                outputs = model(x=x, t=t, lengths=lengths)

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        # 打印统计信息
        logPrint(f'Epoch {epoch + 1}/{num_epochs}, '
                 f'Train Loss: {train_loss / len(train_loader):.4f}, '
                 f'Train Acc: {train_acc:.2f}%, '
                 f'Val Loss: {val_loss / len(val_loader):.4f}, '
                 f'Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val[1]:
            best_val = [train_acc, val_acc]
            torch.save(
                model.state_dict(),
                config.modelPath / f'{config.model}_{dataset.validSpeaker}.pth'
            )

        if train_acc > best_train[0]:
            best_train = [train_acc, val_acc]

    logPrint(f'Training complete.')
    logPrint(f'best on train: train_acc={best_train[0]}, val_acc={best_train[1]}')
    logPrint(f'best on val: train_acc={best_val[0]}, val_acc={best_val[1]}')


if __name__ == '__main__':
    logPrint(clear=True)
    data = DataSet(config.dataset)
    data.setValidSpeaker()
    train_deep_model(
        dataset=data, model=config.model,
        num_epochs=config.epochs, batch_size=config.batchSize,
        learning_rate=config.lr, dropout=config.dropout
    )
    # for i in range(4):
    #     data.setValidSpeaker(i)
    #     train_deep_model(data)
