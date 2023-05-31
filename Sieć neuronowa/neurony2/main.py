import numpy as np

def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, n_inputs, bias=0., weights=None):
        self.b = bias
        if weights:
            self.ws = np.array(weights)
        else:
            self.ws = np.random.rand(n_inputs)

    def _f(self, x):  # activation function (here: leaky_relu)
        return max(x * .1, x)

    def __call__(self, xs):  # calculate the neuron's output
        return self._f(xs @ self.ws + self.b)

class NeuralNetwork:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.hidden_weights1 = None
        self.hidden_weights2 = None
        self.output_weights = None

    def forward(self, input_data):
        hidden1_outputs = [neuron(input_data) for neuron in self.hidden_layer1]
        hidden2_outputs = [neuron(hidden1_outputs) for neuron in self.hidden_layer2]
        output = self.output_neuron(hidden2_outputs)
        return output

    def train(self, X, y, num_iterations):
        self.hidden_weights1 = 2 * np.random.random((X.shape[1] + 1, 4)) - 1
        self.hidden_weights2 = 2 * np.random.random((4 + 1, 4)) - 1
        self.output_weights = 2 * np.random.random((4 + 1, y.shape[1])) - 1

        for i in range(num_iterations):
            input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))
            hidden_layer1_outputs = np.hstack((np.ones((X.shape[0], 1)), sigmoid(np.dot(input_layer_outputs, self.hidden_weights1))))
            hidden_layer2_outputs = np.hstack((np.ones((X.shape[0], 1)), sigmoid(np.dot(hidden_layer1_outputs, self.hidden_weights2))))
            output_layer_outputs = np.dot(hidden_layer2_outputs, self.output_weights)

            output_error = output_layer_outputs - y
            hidden_error2 = hidden_layer2_outputs[:, 1:] * (1 - hidden_layer2_outputs[:, 1:]) * np.dot(output_error, self.output_weights.T[:, 1:])
            hidden_error1 = hidden_layer1_outputs[:, 1:] * (1 - hidden_layer1_outputs[:, 1:]) * np.dot(hidden_error2, self.hidden_weights2.T[:, 1:])

            hidden_pd2 = hidden_layer1_outputs[:, :, np.newaxis] * hidden_error2[:, np.newaxis, :]
            hidden_pd1 = input_layer_outputs[:, :, np.newaxis] * hidden_error1[:, np.newaxis, :]
            output_pd = hidden_layer2_outputs[:, :, np.newaxis] * output_error[:, np.newaxis, :]

            total_hidden_gradient2 = np.average(hidden_pd2, axis=0)
            total_hidden_gradient1 = np.average(hidden_pd1, axis=0)
            total_output_gradient = np.average(output_pd, axis=0)

            self.hidden_weights2 += -self.alpha * total_hidden_gradient2
            self.hidden_weights1 += -self.alpha * total_hidden_gradient1
            self.output_weights += -self.alpha * total_output_gradient

    def predict(self, X):
        input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))
        hidden_layer1_outputs = np.hstack((np.ones((X.shape[0], 1)), sigmoid(np.dot(input_layer_outputs, self.hidden_weights1))))
        hidden_layer2_outputs = np.hstack((np.ones((X.shape[0], 1)), sigmoid(np.dot(hidden_layer1_outputs, self.hidden_weights2))))
        output_layer_outputs = np.dot(hidden_layer2_outputs, self.output_weights)
        return output_layer_outputs

neuron = Neuron(3)
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
])

y = np.array([[0, 1, 0, 1, 1, 0]]).T

network = NeuralNetwork(alpha=0.1)

network.train(X, y, num_iterations=10000)

print("Output After Training: \n{}".format(network.predict(X)))
