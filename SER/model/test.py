from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import BiLSTM, MLP, CombineModel
from dataset import DataSet


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


def main():
    zoo = Path('f:/ASR/SER/zoo')
    DATASET = 'SAVEE'
    noise = 'clean'
    result = pd.DataFrame(
        columns=['delta', 'mfcc=13', 'mfcc=26', 'mfcc=39'],
        data=[['1', '0', '0', '0'], ['2', '0', '0', '0'], ['3', '0', '0', '0']]
    )
    result.set_index('delta', inplace=True)
    print(result)
    for mfcc in [13, 26, 39]:
        for delta in [1, 2, 3]:
            test_acc = []
            for MODEL in ['lstm', 'mlp', 'combine']:
                if MODEL == 'lstm':
                    model = BiLSTM(input_size=mfcc * delta)
                elif MODEL == 'mlp':
                    model = MLP()
                elif MODEL == 'combine':
                    model = CombineModel(temporal_dim=mfcc * delta)
                else:
                    continue
                model.load_state_dict(torch.load(
                    zoo / f'{DATASET}_{noise}_{mfcc}_{delta}' / f'{MODEL}.pth'
                ))
                dataset = f'{DATASET}_{noise}_{mfcc}_{delta}'
                acc = test(model, DataSet(dataset, mfcc * delta))
                test_acc.append(acc)
            test_acc = f'{test_acc[0]:.2f}-{test_acc[1]:.2f}-{test_acc[2]:.2f}'
            result.loc[f'{delta}', f'mfcc={mfcc}'] = test_acc
    print(result)
    result.to_csv(f'{DATASET}_{noise}.csv')


if __name__ == '__main__':
    main()
