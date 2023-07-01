class ProphetLinearRegression:
    def __init__(self):
        self.m = 0
        self.c = 0

    def fit(self, X, y):
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
        return self.m * X + self.c

    def mse(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()
