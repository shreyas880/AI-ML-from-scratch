import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from naive_bayes import NaiveBayes
import time


def accuracy(y_pred, y_test):
    return np.sum(y_test == y_pred)/len(y_test)

start = time.time()

X, y = datasets.make_classification(n_samples=2000, n_features=15, n_classes=2, random_state=1234)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = NaiveBayes()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print(accuracy(y_pred, y_test))
end = time.time()
print("Time taken: ",end - start)