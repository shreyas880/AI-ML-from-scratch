import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from forest import RandomForest

ds = datasets.load_breast_cancer()
X, y = ds.data, ds.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = RandomForest(num_trees=20)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred)/len(y_test)

acc = accuracy(y_test, predictions)

print(acc)