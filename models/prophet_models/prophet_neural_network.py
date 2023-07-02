import numpy as np


class ProphetNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.weights1 = np.random.normal(0, np.sqrt(2. / (input_dim + hidden_dim)), (input_dim, hidden_dim))
        self.weights2 = np.random.normal(0, np.sqrt(2. / (hidden_dim + output_dim)), (hidden_dim, output_dim))
        self.bias1 = np.zeros((1, hidden_dim))
        self.bias2 = np.zeros((1, output_dim))
        self.learning_rate = learning_rate

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x >= 0, x, x * alpha)

    def leaky_relu_derivative(self, x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.layer1 = self.leaky_relu(self.z1)
        self.z2 = np.dot(self.layer1, self.weights2) + self.bias2
        output = self.leaky_relu(self.z2)
        return output

    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.leaky_relu_derivative(output)

        self.layer1_error = self.output_delta.dot(self.weights2.T)
        self.layer1_delta = self.layer1_error * self.leaky_relu_derivative(self.layer1)

        self.weights2 += self.learning_rate * np.dot(self.layer1.T, self.output_delta) / X.shape[0]
        self.bias2 += self.learning_rate * np.sum(self.output_delta, axis=0) / X.shape[0]

        self.weights1 += self.learning_rate * np.dot(X.T, self.layer1_delta) / X.shape[0]
        self.bias1 += self.learning_rate * np.sum(self.layer1_delta, axis=0) / X.shape[0]

    def fit(self, X, y, epochs=0):
        if epochs == 0:
            epochs = len(y) * 3
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)
