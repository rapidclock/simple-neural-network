from data_prep import process_csv
from nn.model import NeuralNetwork
from nn.layers import InputLayer, Dense
from nn.loss import CrossEntropy
from nn.optimizer import SGD
from nn.activations import sigmoid, tanh

test_file = '../data/mnist_test.csv'
train_file = '../data/mnist_train.csv'

x_train, y_train = process_csv(train_file)
x_test, y_test = process_csv(test_file)

model = NeuralNetwork()

model.addLayer(InputLayer((1, 784)))
model.addLayer(Dense(neuron_count=300, activation=tanh()))
model.addLayer(Dense(neuron_count=10, activation=sigmoid()))

model.compile(loss=CrossEntropy(), optimizer=SGD(alpha=0.000006))

train_loss, train_acc, val_loss, val_acc = model.fit(x_test, y_test, validation_set=True,
                                                     validation_split=0.1, epochs=1, batch_size=100)