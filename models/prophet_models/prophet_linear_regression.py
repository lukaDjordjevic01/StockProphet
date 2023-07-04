class ProphetLinearRegression:
    def __init__(self):
        """
        This class represents a simple linear regression model. It's initialized with slope (m) and intercept (c)
        both set to 0.
        """
        self.m = 0
        self.c = 0

    def fit(self, X, y):
        """
        This function fits the linear regression model on the input data. It calculates the slope (m) and intercept (c)
        of the line that minimizes the sum of squared residuals between the observed and predicted target values.
        It assumes that the input features X are one-dimensional.

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
        This function predicts the target value for each instance in the input dataset using the equation of the line
        y = mx + c, where m is the slope and c is the intercept. It assumes that the input features X are one-dimensional.

        :param X:
        :return:
        """
        return self.m * X.flatten() + self.c

    def mse(self, y_true, y_pred):
        """
        This function calculates the mean squared error (MSE) between the observed target values (y_true) and
        the predicted target values (y_pred). The MSE is a measure of the model's prediction error.

        :param y_true:
        :param y_pred:
        :return:
        """
        return ((y_true - y_pred) ** 2).mean()
