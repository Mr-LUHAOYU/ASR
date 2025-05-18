import joblib
import pandas as pd
import numpy as np
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
    def __init__(self, name: str):
        self.name = name
        self.path = config.basePath / 'features' / self.name
        self.speakers = pd.read_csv(self.path / 'speakers.csv').to_numpy().squeeze()
        print('total number of speakers: %d' % len(self.speakers))
        self.validSpeaker = None
        self.scaler = RobustScaler()

    def __len__(self):
        return len(self.speakers)

    def setValidSpeaker(self, validSpeakerID=None):
        if validSpeakerID is None:
            validSpeakerID = np.random.randint(0, self.speakers.shape).item()
        self.validSpeaker = self.speakers[validSpeakerID]
        print(self.validSpeaker, 'set as valid speaker')

    def trainData(self):
        X, y, T = [], [], []
        for speaker in self.speakers:
            if speaker == self.validSpeaker:
                continue

            s = Speaker(self.path, speaker)
            X.append(s.X)
            y.append(s.y)
            T.append(s.T)

        X = pd.concat(X)
        y = pd.concat(y)
        T = pd.concat(T)

        return self._process(X, y, T)

    def validData(self):
        s = Speaker(self.path, self.validSpeaker)
        return self._process(s.X, s.y, s.T)

    def _process(self, X, y, T):
        if hasattr(self.scaler, 'mean_'):  # 检查是否已经fit过
            X = self.scaler.transform(X)  # 直接transform
        else:
            X = self.scaler.fit_transform(X)  # 首次需要fit_transform
            joblib.dump(self.scaler, "robust_scaler.pkl")

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(config.encoder.transform(y.squeeze()))
        T = torch.tensor(T.to_numpy(), dtype=torch.float)
        N, D = T.shape
        T = T.reshape(N // 117, 117, D).transpose(1, 2)
        lengths = (~torch.isnan(T[:, :, 0])).sum(dim=1)
        return X, y, T, lengths


if __name__ == '__main__':
    dataset = DataSet(config.dataset)
    print(len(dataset))
