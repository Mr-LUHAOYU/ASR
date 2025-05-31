import os

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from models import BiLSTM, MLP, CombineModel
from dataset import DataSet
from configs import config

torch.random.manual_seed(42)


def logPrint(*args, **kwargs):
    if kwargs.get('clear', False):
        file = open(config.trainLog, 'w')
        print(file=file, end='', flush=True)
        file.close()
    else:
        file = open(config.trainLog, 'a')
        print(*args, **kwargs)
        print(*args, **kwargs, file=file)
        file.close()


def train_deep_model(
        datasets: DataSet, model: str = 'lstm',
        num_epochs=10, batch_size=32, learning_rate=0.001,
        mfcc=13
):
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model.lower() == 'combine':
        model = CombineModel(temporal_dim=mfcc).to(device)
    elif model.lower() == 'lstm':
        model = BiLSTM(input_size=mfcc).to(device)
    elif model.lower() == 'mlp':
        model = MLP().to(device)
    else:
        raise Exception('Unknown model')

    X, y, T, lengths = datasets.data('train')
    train_dataset = TensorDataset(X, y, T, lengths)

    X, y, T, lengths = datasets.data('val')
    val_dataset = TensorDataset(X, y, T, lengths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    best_train = [0.0, 0.0]
    best_val = [0.0, 0.0]

    for epoch in range(num_epochs):
        train_acc, val_acc, train_loss, val_loss = train_epoch(
            model=model, train_loader=train_loader, val_loader=val_loader,
            device=device, criterion=criterion, optimizer=optimizer
        )

        # 打印统计信息
        logPrint(f'Epoch {epoch + 1}/{num_epochs}, '
                 f'Train Loss: {train_loss:.4f}, '
                 f'Train Acc: {train_acc:.2f}%, '
                 f'Val Loss: {val_loss:.4f}, '
                 f'Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val[1]:
            best_val = [train_acc, val_acc]
            torch.save(
                model.state_dict(),
                config.modelPath / config.dataset / f'{config.savename}.pth'
            )

        if train_acc > best_train[0]:
            best_train = [train_acc, val_acc]


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


def train_epoch(
        train_loader, val_loader, model,
        device, criterion, optimizer
):
    train_loss, train_acc = run(
        dataloader=train_loader, model=model,
        criterion=criterion, optimizer=optimizer, device=device,
        options='train'
    )

    val_loss, val_acc = run(
        dataloader=val_loader, model=model,
        criterion=criterion, optimizer=optimizer, device=device,
        options='val'
    )

    return train_acc, val_acc, train_loss, val_loss


def main():
    logPrint(clear=True)
    os.makedirs(config.modelPath / config.dataset, exist_ok=True)
    datasets = DataSet(config.dataset, config.mfcc)
    train_deep_model(
        datasets=datasets,
        model=config.model, num_epochs=config.epochs,
        batch_size=config.batchSize, learning_rate=config.lr,
        mfcc=config.mfcc
    )


if __name__ == '__main__':
    main()
