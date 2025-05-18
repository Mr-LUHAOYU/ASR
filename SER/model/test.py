from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import BiLSTM, MLP, CombineModel
from dataset import DataSet


def test(
    model, dataset: DataSet, batch_size: int = 64
):
    model.eval()

    X, y, T, lengths = dataset.trainData()
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

    test_acc = test_correct / test_total

    return test_acc


def main():
    modelSAVEE = []
    modelEmoDB = []
    zoo = Path('f:/ASR/SER/zoo')
    for MODEL in [BiLSTM, MLP, CombineModel]:
        model1 = MODEL()
        model1.load_state_dict(
            torch.load(zoo / 'SAVEE' / f'{model1.name}.pth')
        )
        modelSAVEE.append(model1)

        model2 = MODEL()
        model2.load_state_dict(
            torch.load(zoo / 'EmoDB' / f'{model2.name}.pth')
        )
        modelEmoDB.append(model2)

    dataset = DataSet('test')
    accSAVEE = [test(model, dataset) for model in modelSAVEE]
    accEmoDB = [test(model, dataset) for model in modelEmoDB]

    print(accSAVEE)
    print(accEmoDB)


if __name__ == '__main__':
    main()