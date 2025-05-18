from sklearn.ensemble import RandomForestClassifier
from dataset import DataSet
import numpy as np


data = DataSet('SAVEE')

accuracies = []
for speakerID in range(len(data)):
    data.setValidSpeaker(speakerID)

    X_train, y_train, _, _ = data.trainData()
    X_test, y_test, _, _ = data.validData()
    X_train = X_train.numpy()
    X_test = X_test.numpy()
    y_train = y_train.numpy()
    y_test = y_test.numpy()

    model = RandomForestClassifier()

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    accuracies.append(acc)

print(f"\nOverall LOSO Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
