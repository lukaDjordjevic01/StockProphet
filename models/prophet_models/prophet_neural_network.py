import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


class ProphetNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.1, l2_reg=0.001):
        self.weights1 = np.random.normal(0, np.sqrt(2. / (input_dim + hidden_dim)), (input_dim, hidden_dim))
        self.weights2 = np.random.normal(0, np.sqrt(2. / (hidden_dim + output_dim)), (hidden_dim, output_dim))
        self.bias1 = np.zeros((1, hidden_dim))
        self.bias2 = np.zeros((1, output_dim))
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.scaler = StandardScaler()

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

        self.weights2 += self.learning_rate * (
                np.dot(self.layer1.T, self.output_delta) / X.shape[0] - self.l2_reg * self.weights2)
        self.bias2 += self.learning_rate * np.sum(self.output_delta, axis=0) / X.shape[0]

        self.weights1 += self.learning_rate * (
                np.dot(X.T, self.layer1_delta) / X.shape[0] - self.l2_reg * self.weights1)
        self.bias1 += self.learning_rate * np.sum(self.layer1_delta, axis=0) / X.shape[0]

    def fit(self, X, y, epochs=100, batch_size=128):  # Added batch_size parameter
        X = self.scaler.fit_transform(X)
        for epoch in range(epochs):
            X, y = shuffle(X, y)
            for i in range(0, X.shape[0], batch_size):  # Mini-batch gradient descent
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.forward(X)
