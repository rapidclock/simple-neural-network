import numpy as np


class Loss(object):
    def __init__(self):
        pass

    def compute(self, y, y_cap):
        pass

    def derivative(self, y, y_cap):
        pass


class CrossEntropy(Loss):
    def compute(self, y_true, y_cap):
        return (y_true * np.log2(y_cap+1e-4)) + ((1 - y_true) * np.log2((1 - y_cap+1e-4)))

    def derivative(self, y_true, y_cap):
        return np.divide((y_cap - y_true), (y_cap * (1 - y_cap + 1e-4)))
