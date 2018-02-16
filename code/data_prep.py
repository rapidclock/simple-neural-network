import numpy as np


def process_csv(file, onehot=True):
    data = np.genfromtxt(file, delimiter=',')
    labels = data[:, 0].astype(int)
    data = data[:, 1:]
    if onehot:
        labels = onehot_encode(labels)
    data /= 255
    return data, labels


def onehot_encode(labels, oneof=10):
    l = len(labels)
    output = np.zeros((l, oneof))
    output[np.arange(l), labels] = 1.
    return output
