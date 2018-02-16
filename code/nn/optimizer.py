import numpy as np


class Optimize(object):
    def __init__(self):
        pass

    def compile(self, layers, loss):
        self.layers = layers[1:]
        self.loss = loss

    def optimize(self, data, label):
        pass


class SGD(Optimize):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def optimize(self, data, label):
        op = data
        for layer in self.layers:
            op = layer.get_forward_pass_output(op)
        # total_loss = self.loss.compute(label, op)
        self.backpropogate(label, op)

    def backpropogate(self, y_true, y_cap):
        dLoss = self.loss.derivative(y_true, y_cap)
        for i in range(len(self.layers)-1, 0, -1):
            self.layers[i]._backward_pass(dLoss)
            self.layers[i].weights -= self.alpha * self.layers[i].dL_dW
            self.layers[i].x -= self.alpha * self.layers[i].dL_dX
            dLoss = self.layers[i].x
        self.layers[0]._backward_pass(dLoss)
        self.layers[0].weights -= self.alpha * self.layers[0].dL_dW
