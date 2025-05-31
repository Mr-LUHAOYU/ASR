import argparse
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basePath', type=str, default='f:/ASR/SER', help='absolute path')
    parser.add_argument('--trainLogfile', type=str, default='train.log')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--batchSize', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--model', type=str, default='combine', choices=['combine', 'lstm', 'mlp'], help='model type')
    parser.add_argument('--modelPath', type=str, default='zoo', help='model path')
    parser.add_argument('--savename', type=str, default=None)
    parser.add_argument('--mfcc', type=int, default=13)
    parser.add_argument('--dataset', type=str, default='SAVEE')
    args = parser.parse_args()
    return args


class Config(object):
    def __init__(self):
        ...

    def loadParams(self, args):
        self.basePath = args.basePath
        self.basePath = Path(self.basePath)
        self.trainLog = args.trainLogfile
        self.epochs = args.epochs
        self.batchSize = args.batchSize
        self.lr = args.lr
        self.dropout = args.dropout
        self.model = args.model
        self.modelPath = self.basePath / args.modelPath
        self.modelPath.mkdir(parents=True, exist_ok=True)
        self.mfcc = args.mfcc
        self.emotion = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.encoder = {e: i for i, e in enumerate(self.emotion)}
        self.encoder['anger'] = 4

        self.savename = args.savename
        if self.savename is None:
            self.savename = self.model
        self.dataset = args.dataset


config = Config()
config.loadParams(
    get_args()
)
