import numpy as np


class Layer(object):
    def __init__(self, neuron_count, activation):
        self.weights = None
        self.x = None
        self.neurons = neuron_count
        self.activation = activation
        self.intermediate = None
        self.output = None
        self.next_layer = None

    def run(self):
        pass

    def _forward_pass(self, X):
        pass

    def initialize_weights_and_bias(self, input_size):
        pass

    def _backward_pass(self, dLoss, alpha):
        pass

    def getString(self):
        return 'Layer'


class InputLayer(Layer):
    def __init__(self, shape):
        super().__init__(0, None)
        self.shape = shape

    def getString(self):
        return "Input"


class Dense(Layer):
    def __init__(self, neuron_count, activation):
        super().__init__(neuron_count, activation)

    def initialize_weights_and_bias(self, input_size):
        self.weights = np.random.normal(loc=0.0, scale=pow(input_size, -0.05),size=(self.neurons, input_size))
        self.bias = 0.

    def _forward_pass(self, X):
        self.x = X
        self.intermediate = np.dot(self.x, self.weights.T)
        self.output = self.activation.compute(self.intermediate)
        self.dInterOp = self.activation.derivative(self.intermediate)
        self.dXInter = self.weights
        self.dWInter = self.x

    def get_forward_pass_output(self, X):
        self._forward_pass(X)
        return self.output

    def get_weights(self):
        return self.weights

    def backprop(self, dLoss):
        self._backward_pass(dLoss)
        # return self.x

    def _backward_pass(self, dLoss):
        # print(dLoss.shape, self.dInterOp.shape)
        self.dL_dInter = dLoss * self.dInterOp
        self.dL_dW = np.dot(self.dL_dInter.T, self.dWInter)
        self.dL_dX = np.dot(self.dL_dInter, self.dXInter)
        # self.weights -= alpha * dL_dW
        # self.x -= alpha * dL_dX

    def update_weights(self, updated_weights):
        self.weights = updated_weights

    def getString(self):
        return "Dense"
