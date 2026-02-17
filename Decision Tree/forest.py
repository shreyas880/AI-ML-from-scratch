import numpy as np
from dt import DecisionTree
from collections import Counter

class RandomForest:
    def __init__(self, num_trees = 10, min_samples_split = 2, max_depth = 100,  num_features = None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.num_trees):
            tree = DecisionTree(self.min_samples_split, self.max_depth, self.num_features)

            X_sample, y_sample = self._samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)


    def predict(self, X_test):
        predictions = np.array([tree.predict(X_test) for tree in self.trees])

        tree_predictions = np.swapaxes(predictions, 0, 1)

        return np.array([self._most_common(pred) for pred in tree_predictions])

    def _most_common(self, y):
        return Counter(y).most_common(1)[0][0]

    def _samples(self, X, y):
        num_samples = X.shape[0]
        idxs = np.random.choice(num_samples, num_samples, replace=True)
        return X[idxs], y[idxs]