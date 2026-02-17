import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN

# Colours maps to represent the data points with different labels
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Loading the dataset
iris = datasets.load_iris()
x, y = iris.data, iris.target

x_train, x_test , y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)


# plt.figure()
# plt.scatter(x[:, 2], x[:, 3], c=y, cmap=cmap, edgecolors='k', s=20)
# plt.show()


# Creating an instance of the classifier, although, this time with k=5
clf = KNN(k=5)

# Fitting training data
clf.fit(x_train, y_train)

# Getting Predictions and storing it in an array
predictions = clf.predict(x_test)

# Primitive method of measuring accuracy by looping and checking through each iteration to see if the prediction is same as the actual value
count = 0
# for i in range(len(predictions)):
#     if predictions[i] == y_test[i]:
#         count += 1 

# Alternative implementation of the above loop can be done by the following:
count = np.sum(predictions == y_test)

accuracy = count/len(y_test)

print(count)
print(accuracy)