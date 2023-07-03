class ProphetLinearRegression:
    def __init__(self):
        self.m = 0
        self.c = 0

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        X = X.flatten()
        x_mean = X.mean()
        y_mean = y.mean()

        num = 0
        den = 0
        for i in range(len(X)):
            num += (X[i] - x_mean) * (y[i] - y_mean)
            den += (X[i] - x_mean) ** 2

        self.m = num / den
        self.c = y_mean - self.m * x_mean

    def predict(self, X):
        """

        :param X:
        :return:
        """
        return self.m * X.flatten() + self.c

    def mse(self, y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        return ((y_true - y_pred) ** 2).mean()
