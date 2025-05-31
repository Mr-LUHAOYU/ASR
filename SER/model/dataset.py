# import joblib
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from configs import config


class Speaker(object):
    def __init__(self, path, speaker):
        """
        :param path: base path, like features/SAVEE
        """
        self.path = path
        self.speaker = speaker
        self.X = []
        self.y = []
        self.T = []
        self._extractFeatures()

    def _extractFeatures(self):
        self.X = pd.read_csv(self.path / (self.speaker + 'X.csv'))
        self.y = pd.read_csv(self.path / (self.speaker + 'y.csv'))
        self.T = pd.read_csv(self.path / (self.speaker + 'T.csv'))


class DataSet(object):
    def __init__(self, name: str, mfcc):
        self.name = name
        self.path = config.basePath / 'features' / self.name
        self.mfcc = mfcc
        self.scaler = RobustScaler()
        # self.zoo = config.modelPath / config.dataset

    def data(self, opt, msg: bool = True):
        path = self.path / opt
        speakers = pd.read_csv(path / 'speakers.csv').to_numpy().squeeze().tolist()
        if isinstance(speakers, str):
            speakers = [speakers]
        msg and print(f'total {opt} speakers: {len(speakers)}')

        X, y, T = [], [], []
        for speaker in speakers:
            s = Speaker(path, speaker)
            X.append(s.X)
            y.append(s.y)
            T.append(s.T)

        X = pd.concat(X)
        y = pd.concat(y)
        T = pd.concat(T)

        return self._process(X, y, T)

    def _process(self, X: pd.DataFrame, y: pd.DataFrame, T):
        # if hasattr(self.scaler, 'mean_'):  # 检查是否已经fit过
        #     X = self.scaler.transform(X)  # 直接transform
        # else:
        #     X = self.scaler.fit_transform(X)  # 首次需要fit_transform
        #     joblib.dump(self.scaler, self.zoo / "scaler.pkl")

        y['label'] = y['label'].map(config.encoder)
        X = torch.tensor(X.to_numpy(), dtype=torch.float)
        y = torch.tensor(y.to_numpy().squeeze(), dtype=torch.long)
        T = torch.tensor(T.to_numpy(), dtype=torch.float)
        N, D = T.shape
        T = T.reshape(N // self.mfcc, self.mfcc, D).transpose(1, 2)
        lengths = (~torch.isnan(T[:, :, 0])).sum(dim=1)

        return X, y, T, lengths


if __name__ == '__main__':
    dataset = DataSet(config.dataset, mfcc=13)
