import numpy as np


class ProphetLinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        """
        This class represents a simple Linear Support Vector Machine model.
        The learning_rate sets the step size for the gradient descent, the lambda_param is the regularization parameter,
        and n_iters is the number of iterations for the gradient descent.

        :param learning_rate:
        :param lambda_param:
        :param n_iters:
        """

        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        This function fits the SVM model on the input data. It uses gradient descent to minimize the hinge loss
        function and find the optimal values for the weights and bias. It also converts the class labels to -1 and 1
        (this is needed for the calculation of the hinge loss).

        :param X:
        :param y:
        :return:
        """

        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        """
        This function predicts the class label for each instance in the input dataset based on the learned weights and
        bias. If the linear output (X * w - b) is positive, it assigns the instance to class 1, otherwise it assigns it
        to class -1.
        
        :param X:
        :return:
        """
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
