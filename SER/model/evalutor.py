import librosa
import numpy as np
import torch
from models import BiLSTM, MLP, CombineModel
from pathlib import Path
import opensmile
from sklearn.preprocessing import RobustScaler


class Extractor(object):
    def __init__(
            self,
            smile: opensmile.Smile,
            scaler: RobustScaler,
            n_mfcc=39
    ):
        self.smile = smile
        self.n_mfcc = n_mfcc
        self.scaler = scaler

    def __call__(self, audio, sr):
        return self.extract_features(audio, sr)

    def temporal(self, audio, sr):
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        return features

    def features(self, audio, sr):
        return self.smile.process_signal(audio, sr)

    def extract_features(self, audio, sr):
        X = self.features(audio, sr)
        T = self.temporal(audio, sr)
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float)
        T = torch.tensor(T, dtype=torch.float)
        T = T.squeeze(1)
        N, D = T.shape
        T = T.reshape(N // 117, 117, D).transpose(1, 2)
        lengths = (~torch.isnan(T[:, :, 0])).sum(dim=1)
        return X, T, lengths


class Evaluator:
    def __init__(self, model_zoo: Path):
        self.model = None
        self.model_zoo = model_zoo

    def init_model(self):
        self.model = CombineModel()
        self.model.load_state_dict(torch.load(self.model_zoo / 'combine.pth'))

    def set_model(self, model: str):
        if self.model.name == model.lower():
            return

        if model.lower() == 'lstm':
            self.model = BiLSTM()
            self.model.load_state_dict(torch.load(self.model_zoo / 'lstm.pth'))
        elif self.model.lower() == 'mlp':
            self.model = MLP()
            self.model.load_state_dict(torch.load(self.model_zoo / 'mlp.pth'))
        elif self.model.lower() == 'combine':
            self.model = CombineModel()
            self.model.load_state_dict(torch.load(self.model_zoo / 'combine.pth'))
        else:
            raise Exception('Unknown model')

    def evaluate(self, data):
        self.model.eval()
        return self.model.eval(data)

    def __call__(self, data):
        return self.evaluate(data)
