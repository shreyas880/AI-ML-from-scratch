import numpy as np

# Logic:
# Given a training set, plot a line by initialising parameters to some value
# calculate the erorr, and then gradient descent tells us a new value for the parameters
# Re-calculate error with these values and repeat until error is low enough
# Line has been achieved
# Check with testing set now and calculate the errors
# Keep a good learning rate (depends on dataset)

class LinearRegression:
    def __init__(self, lr = 0.01, iterations = 1000):
        self.lr = lr
        self.iterations = iterations
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        # Initialization happens here:
        # X.shape() returns the number of samples first and then number of features
        num_samples, num_features = X.shape

        # The number of weights is initialised to a zero matrix of the dimension of the number of features
        self.weight = np.zeros(num_features)

        # Bias is initialised to zero
        self.bias = 0

        # Iterations and fine tuning happens here:
        for i in range(self.iterations):
            # We need to get a "predicted" y value now, so that we can get the error and then fine tune the weights and biases
            # y = wx + b
            y_pred = np.dot(X, self.weight) + self.bias

            # Error function is the mean square error (NOT rmse)
            # dw is the derivative of the error function wrt weight w
            # db is the derivative of the error function wrt bias b
            dw = (1/num_samples) * np.dot(np.transpose(X), (y_pred - y))
            db = (1/num_samples) * np.sum(y_pred - y)

            # Adjusting the weights according to error and learning rate
            self.weight -= self.lr*dw
            self.bias -= self.lr*db

    def predict(self, X):
        y_pred = np.dot(X, self.weight) + self.bias

        return y_pred
