import numpy as np
import math
from .exceptions import InvalidInputLayerException


class BaseNeuralNetwork(object):
    """Base Class for all Neural Networks
    """

    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def addLayer(self, layer):
        pass

    def compile(self, loss, optimizer=None):
        self.loss = loss
        self.optimizer = optimizer
        self._check_input_layer()
        self.optimizer.compile(self.layers, self.loss)

    def _check_input_layer(self):
        print(self.layers[0].getString())
        if self.layers[0].getString() != "Input":
            raise InvalidInputLayerException("No Input Layer!")

    def fit(self, x_train, y_train, epochs=1, batches=1):
        pass

    def predict(self, x_test):
        pass

    def accuracy(self, y_true, y_pred):
        pass

    def _validate(self):
        pass


class NeuralNetwork(BaseNeuralNetwork):
    def __init__(self):
        super().__init__()
        self.inputsize = 0
        self.data_size = 0
        self.validation_split = 0
        self.batches = 0
        self.epochs = 0
        self.X = None
        self.Y = None

    def addLayer(self, layer):
        self.layers.append(layer)

    def fit(self, x_train, y_train, validation_set=False, validation_split=0., epochs=1, batch_size=1):
        self.X = x_train
        self.Y = y_train
        self.batches = batch_size
        self.epochs = epochs
        self.data_size = len(self.Y)
        self.validation = validation_set
        if self.validation:
            self.validation_split = validation_split
            self.split_validation_data()
        self.init_weights()
        self._make_minibatches()
        return self.start_training()

    def split_validation_data(self):
        self.validation_size = math.floor(self.validation_split * self.data_size)
        self.val_X, self.val_Y = self.X[-self.validation_size:], self.Y[-self.validation_size:]
        self.X, self.Y = self.X[:-self.validation_size], self.Y[:-self.validation_size]

    def _make_minibatches(self):
        remainder = self.data_size % self.batches
        q = self.data_size // self.batches
        if remainder == 0:
            self.mb_x, self.mb_y = np.array(np.vsplit(self.X, q)), np.array(np.vsplit(self.Y, q))
        else:
            last_batch_x, last_batch_y = self.X[-remainder:], self.Y[-remainder:]
            self.mb_x, self.mb_y = np.vsplit(self.X[:-remainder], q), np.vsplit(self.Y[:-remainder], q)
            self.mb_x.append(last_batch_x)
            self.mb_y.append(last_batch_y)
            self.mb_x, self.mb_y = np.array(self.mb_x), np.array(self.mb_y)

    def init_weights(self):
        _, input_size = self.layers[0].shape
        for layer in self.layers[1:]:
            layer.initialize_weights_and_bias(input_size)
            input_size = layer.neurons

    def _validate(self):
        return self.accuracy(self._network_forward(self.val_X), self.val_Y)

    def _network_forward(self, x):
        op = x
        for layer in self.layers[1:]:
            op = layer.get_forward_pass_output(op)
        return op

    def predict(self, x_test):
        return self._network_forward(self.val_X)

    def test(self, test_x, test_y):
        y_pred = self._network_forward(test_x)
        return self.loss.compute(test_y, y_pred), self.accuracy(y_pred, test_y)

    def start_training(self):
        validation_loss = []
        validation_accuracy = []
        testing_loss = []
        testing_accuracy = []
        for e in range(self.epochs):
            for i in range(len(self.mb_x)):
                minibatch_x, minibatch_y = self.mb_x[i], self.mb_y[i]
                self.optimizer.optimize(minibatch_x, minibatch_y)
            # print(self.val_X.shape, self.val_Y.shape)
            if self.validation:
                vloss, vacc = self.test(self.val_X, self.val_Y)
                validation_accuracy.append(vacc)
                validation_loss.append(vloss)
            trloss, tracc = self.test(self.X, self.Y)
            testing_accuracy.append(tracc)
            testing_loss.append(trloss)
        return np.array(testing_loss), np.array(testing_accuracy), np.array(validation_loss), np.array(
            validation_accuracy)

    def accuracy(self, y_true, y_pred):
        y_t = np.argmax(y_true, axis=1)
        y_p = np.argmax(y_pred, axis=1)
        return np.sum(y_t == y_p) / len(y_true)
