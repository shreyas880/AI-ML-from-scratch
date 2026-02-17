import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_pred)

ds = datasets.load_breast_cancer()
X, y = ds.data, ds.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy(y_pred, y_test)
print(acc)