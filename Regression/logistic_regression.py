import numpy as np

def sigmoid(term):
    return 1/(1 + np.exp(-term))

class LogisticRegression:
    def __init__(self, lr=0.01, iterations = 1000):
        self.weight = None
        self.bias = None
        self.iterations = iterations
        self.lr = lr

    def fit(self, X, y):
        pass
        num_samples, num_features = X.shape

        self.weight = np.zeros(num_features)
        self.bias = 0

        for i in range(self.iterations):
            # Got the predictions
            term = np.dot(X, self.weight) + self.bias
            y_pred = sigmoid(term)

            # Need to find errors now
            dw = (1/num_samples)*(np.dot(X.T, y_pred - y))
            db = (1/num_samples)*(np.sum(y_pred - y))

            self.weight -= self.lr*dw
            self.bias -= self.lr*db

    def predict(self, X_test):
        term = np.dot(X_test, self.weight) + self.bias
        y_pred = sigmoid(term)
        label_predictions = [0 if y < 0.5 else 1 for y in y_pred]

        return label_predictions