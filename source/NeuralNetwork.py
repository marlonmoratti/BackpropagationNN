from data import Dataset, DataLoader
import functional as F
import numpy as np

class NeuralNetwork:
    def __init__(self,
                 hidden_layer_sizes=(100,),
                 batch_size=32,
                 learning_rate=1e-3,
                 max_iter=200,
                 shuffle=True,
                 random_state=None):
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random = (
            random_state
            if isinstance(random_state, np.random.RandomState) else
            np.random.RandomState(random_state)
        )

        self.layer_weights = []
        self.layer_activations = []
        self.history = {'epoch': [], 'loss': []}

        self.activation = F.sigmoid
        self.criterion = F.log_loss

    def fit(self, x, y):
        self._initialize_weights(x.shape[1], y.shape[1])

        self.history = {'epoch': [], 'loss': []}

        dataloader = DataLoader(Dataset(x, y), self.batch_size, self.shuffle, self.random)
        for epoch in range(self.max_iter):
            loss = self._train_epoch(dataloader)

            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(loss)

        return self

    def predict(self, x):
        return self._forward(x)

    def _initialize_weights(self, input_size, output_size):
        layer_sizes = [input_size, *self.hidden_layer_sizes, output_size]
        self.layer_weights = [
            self.random.randn(layer_sizes[i - 1] + 1, layer_sizes[i])
            for i in range(1, len(layer_sizes))
        ]

    def _train_epoch(self, dataloader):
        running_loss = 0
        for inputs, targets in dataloader:
            outputs = self._forward(inputs)
            loss = self.criterion(outputs, targets)

            self._backward(targets)

            running_loss += loss * inputs.shape[0]
        
        avg_loss = running_loss / len(dataloader.dataset)
        return avg_loss

    def _forward(self, x):
        self.layer_activations = [x]
        for w in self.layer_weights:
            x = self._add_bias_term(x)
            x = self.activation(np.dot(x, w))
            self.layer_activations.append(x)
        return x

    def _backward(self, targets):
        output_activation = self.layer_activations[-1]
        output_delta = self.criterion(output_activation, targets, True) * self.activation(output_activation, True, True)

        deltas = [output_delta]

        for i in range(len(self.hidden_layer_sizes)):
            layer_activation = self.layer_activations[-(i + 2)]
            weight = self.layer_weights[-(i + 1)]
            delta = np.dot(deltas[i], weight[1:].T) * self.activation(layer_activation, True, True)
            deltas.append(delta)

        for activation, delta, weight in zip(self.layer_activations, deltas[::-1], self.layer_weights):
            activation_with_bias = self._add_bias_term(activation)
            weight -= np.dot(activation_with_bias.T, delta) * self.learning_rate

    def _add_bias_term(self, x):
        ones_column = np.ones((x.shape[0], 1))
        return np.hstack((ones_column, x))
