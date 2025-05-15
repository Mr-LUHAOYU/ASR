import os
import re
import sys
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

sys.path.append('f:/ASR/大作业/model')
from configs.params import Config


class Speaker(object):
    def __init__(self, path):
        self.path = path
        self.X = []
        self.y = []
        self._extractFeatures()

    def _extractFeatures(self):
        for item in os.listdir(self.path):
            full_path = self.path / item
            data = np.load(full_path)['features']

            self.X.append(data)
            letters = re.match(r"^([A-Za-z]+)\d+\.npz$", item).group(1)
            self.y.append(letters)


class DataSet(object):
    def __init__(self, name: str):
        self.name = name
        self.speakers = self._getSpeaker()
        print('total number of speakers: %d' % len(self.speakers))
        self.validSpeaker = None

    def _getSpeaker(self):
        speakers = {}
        for item in os.listdir(Config.basePath / self.name):
            full_path = Config.basePath / self.name / item
            speakers[item] = Speaker(full_path)
        return speakers

    def __len__(self):
        return len(self.speakers)

    def setValidSpeaker(self, validSpeakerID=None):
        speakerList = [key for key, value in self.speakers.items()]
        if validSpeakerID is None:
            validSpeakerID = np.random.randint(0, len(speakerList))
        self.validSpeaker = speakerList[validSpeakerID]
        print(self.validSpeaker, 'set as valid speaker')

    def trainData(self):
        X, y = [], []
        for speakerName, speaker in self.speakers.items():
            if speakerName != self.validSpeaker:
                X.extend(speaker.X)
                y.extend(Config.encoder.transform(speaker.y))
        X = [torch.FloatTensor(x.T) for x in X]
        X = pad_sequence(X, batch_first=True, padding_value=0)
        y = torch.tensor(y)
        lengths = torch.tensor([len(x) for x in X])
        return X, y, lengths

    def validData(self):
        X = self.speakers[self.validSpeaker].X
        y = Config.encoder.transform(self.speakers[self.validSpeaker].y)
        X = [torch.FloatTensor(x.T) for x in X]
        X = pad_sequence(X, batch_first=True, padding_value=0)
        y = torch.tensor(y)
        lengths = torch.tensor([len(x) for x in X])
        return X, y, lengths


if __name__ == '__main__':
    dataset = DataSet('SAVEE')
    print(len(dataset))
