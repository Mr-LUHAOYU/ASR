import pandas as pd
import numpy as np
from params import Config


class Speaker(object):
    def __init__(self, path, speaker):
        """
        :param path: base path, like features/SAVEE
        """
        self.path = path
        self.speaker = speaker
        self.X = []
        self.y = []
        self._extractFeatures()

    def _extractFeatures(self):
        self.X = pd.read_csv(self.path / (self.speaker + 'X.csv'))
        self.y = pd.read_csv(self.path / (self.speaker + 'y.csv'))


class DataSet(object):
    def __init__(self, name: str):
        self.name = name
        self.path = Config.basePath / self.name
        self.speakers = pd.read_csv(self.path / 'speakers.csv').to_numpy().squeeze()
        print('total number of speakers: %d' % len(self.speakers))
        self.validSpeaker = None

    def __len__(self):
        return len(self.speakers)

    def setValidSpeaker(self, validSpeakerID=None):
        if validSpeakerID is None:
            validSpeakerID = np.random.randint(0, self.speakers.shape)
        self.validSpeaker = self.speakers[validSpeakerID].item()
        print(self.validSpeaker, 'set as valid speaker')

    def trainData(self):
        X, y = [], []
        for speaker in self.speakers:
            if speaker == self.validSpeaker:
                continue

            s = Speaker(self.path, speaker)
            X.append(s.X)
            y.append(s.y)
        X = pd.concat(X)
        y = pd.concat(y)
        return X, y

    def validData(self):
        s = Speaker(self.path, self.validSpeaker)
        return s.X, s.y


if __name__ == '__main__':
    dataset = DataSet('SAVEE')
    print(len(dataset))
