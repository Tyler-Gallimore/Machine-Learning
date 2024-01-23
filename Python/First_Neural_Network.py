import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1+ np.exp(-x))
    
    def _sigmoig_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))
    
    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction
    
    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoig_deriv(layer_1)
        dlayer_dbias = 1
        dlayer_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer_dbias
        )

        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer_dweights
        )

        return derror_dbias, derror_dweights
    
    def _update_parameters(self, derror_dbais, derror_dweights):
        self.bias = self.bais - (derror_dbais * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )

learning_rate = 0.1

neural_network = NeuralNetwork(learning_rate)

neural_network.predict(input_vector)