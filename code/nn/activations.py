import numpy as np


class Activation(object):
    def __init__(self):
        pass

    def compute(self, x):
        pass

    def derivative(self, x):
        pass


class sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def compute(self, x):
        return 1. / (1. + np.exp(-x))

    def derivative(self, x):
        inter = self.compute(x)
        return inter * (1 - inter)


class tanh(Activation):
    def compute(self, x):
        return np.tanh(x)

    def derivative(self, x):
        inter = self.compute(x)
        return 1. - np.square(inter)
