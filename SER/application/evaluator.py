import joblib
import librosa
import numpy as np
import torch
from pathlib import Path
import opensmile
from sklearn.preprocessing import RobustScaler, LabelEncoder
from model.models import BiLSTM, MLP, CombineModel


class Extractor(object):
    def __init__(
            self,
            smile: opensmile.Smile | None = None,
            scaler: RobustScaler | None = None,
            n_mfcc=39,
            zoo=None
    ):
        if smile is None:
            smile = opensmile.Smile()
        if scaler is None:
            scaler = joblib.load(zoo / 'robust_scaler.pkl')
        self.smile = smile
        self.n_mfcc = n_mfcc
        self.scaler = scaler

    def __call__(self, audio):
        return self.extract_features(audio)

    def temporal(self, audio):
        y, sr = librosa.load(audio)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        return features

    def features(self, audio):
        return self.smile.process_file(audio)

    def extract_features(self, audio):
        X = self.features(audio)
        T = self.temporal(audio)
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float)
        T = torch.tensor(T, dtype=torch.float)
        T = T.squeeze(1)
        N, D = T.shape
        T = T.reshape(N // 117, 117, D).transpose(1, 2)
        lengths = (~torch.isnan(T[:, :, 0])).sum(dim=1)
        return X, T, lengths


class Evaluator:
    def __init__(self, model_zoo: Path | str, extractor: Extractor | None = None):
        self.model = None
        if isinstance(model_zoo, str):
            model_zoo = Path(model_zoo)
        self.model_zoo = model_zoo
        if extractor is None:
            extractor = Extractor(zoo=model_zoo)
        self.extractor = extractor
        self.init_model()
        self.translate = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']


    def init_model(self):
        self.model = CombineModel()
        self.model.load_state_dict(torch.load(self.model_zoo / 'combine.pth'))

    def set_model(self, model: str):
        if self.model.name == model.lower():
            return

        if model.lower() == 'lstm':
            self.model = BiLSTM()
            self.model.load_state_dict(torch.load(self.model_zoo / 'lstm.pth'))
        elif model.lower() == 'mlp':
            self.model = MLP()
            self.model.load_state_dict(torch.load(self.model_zoo / 'mlp.pth'))
        elif model.lower() == 'combine':
            self.model = CombineModel()
            self.model.load_state_dict(torch.load(self.model_zoo / 'combine.pth'))
        else:
            raise Exception('Unknown model')

    def evaluate(self, audio):
        self.model.eval()
        with torch.no_grad():
            x, t, lengths = self.extractor(audio)
            output = self.model(x=x, t=t, lengths=lengths).squeeze()
            output = torch.argmax(output).item()
            emotion = self.translate[output]
        return emotion

    def __call__(self, audio):
        return self.evaluate(audio)
