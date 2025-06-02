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
    ):
        if smile is None:
            smile = opensmile.Smile()
        self.smile = smile

    def __call__(self, audio, mfcc):
        return self.extract_features(audio, mfcc)

    def temporal(self, audio, mfcc):
        y, sr = librosa.load(audio)
        n_mfcc = int(mfcc[:2])
        delta = int(mfcc[-1])
        if delta == 1:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            return mfcc
        if delta == 2:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            delta_mfcc = librosa.feature.delta(mfcc)
            return np.vstack([mfcc, delta_mfcc])
        if delta == 3:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
            return features
        raise ValueError(f"Unknown delta type {delta}")

    def features(self, audio):
        return self.smile.process_file(audio)

    def extract_features(self, audio, mfcc: str):
        X = self.features(audio)
        T = self.temporal(audio, mfcc)
        # X = self.scaler.transform(X)
        X = torch.tensor(X.to_numpy(), dtype=torch.float)
        T = torch.tensor(T, dtype=torch.float)
        T = T.squeeze(1)
        N, D = T.shape
        mfcc = eval(mfcc)
        T = T.reshape(N // mfcc, mfcc, D).transpose(1, 2)
        lengths = (~torch.isnan(T[:, :, 0])).sum(dim=1)
        return X, T, lengths


class Evaluator:
    def __init__(self, model_zoo: Path | str):
        self.model = None
        self.modelname = None
        self.dataset = None
        self.mfcc = None
        self.noise = None
        if isinstance(model_zoo, str):
            model_zoo = Path(model_zoo)
        self.model_zoo = model_zoo
        self.extractor = Extractor()
        self.translate = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    def set_model(self, model: str, dataset: str, mfcc: str, noise: str):
        if (
            self.modelname == model.lower() and
            self.dataset == dataset.lower() and
            self.mfcc == mfcc and
            self.noise == noise.lower()
        ):
            return

        self.modelname = model.lower()
        self.dataset = dataset.lower()
        self.mfcc = mfcc.lower()
        self.noise = noise.lower()

        if model.lower() == 'lstm':
            self.model = BiLSTM(input_size=eval(mfcc))
        elif model.lower() == 'mlp':
            self.model = MLP()
        elif model.lower() == 'combine':
            self.model = CombineModel(temporal_dim=eval(mfcc))
        else:
            raise Exception('Unknown model')

        mfcc = mfcc.replace('*', '_')
        pth = self.model_zoo / f'{self.dataset}_{noise}_{mfcc}' / f'{model.lower()}.pth'
        self.model.load_state_dict(torch.load(pth))

    def evaluate(self, audio, mfcc):
        self.model.eval()
        with torch.no_grad():
            x, t, lengths = self.extractor(audio, mfcc)
            output = self.model(x=x, t=t, lengths=lengths).squeeze()
            output = torch.argmax(output).item()
            emotion = self.translate[output]
        return emotion

    def __call__(self, audio, mfcc):
        return self.evaluate(audio, mfcc)
