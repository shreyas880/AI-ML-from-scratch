import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold = None, left = None, right = None, *, value=None):
        # Which feature was basis of creation for this node
        self.feature = feature

        # Condition/Threshold for division
        self.threshold = threshold

        # Left child node
        self.left = left

        # Right child node
        self.right = right

        # None for anything that is not a leaf nodes
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, num_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features

        self.root = None
        

    def fit(self, X, y):
        # Making sure the user defined number of features does not exceed the actual number of features in dataset
        # self.num_features is the number of features that the user wants to select for each decision
        self.num_features = X.shape[1] if not self.num_features else min(self.num_features, X.shape[1])

        # Helper function to recursively grow the tree
        self.root = self._grow_tree(X, y, 0)
    
    def _grow_tree(self, X, y, depth):
        # n_feat is not the same as self.num_features as we will be calling this function recursively so the number of features passed will keep decreasing
        n_samples, n_feat = X.shape
        n_labels = len(np.unique(y))

        # Check if max_depth has been reached
        if depth > self.max_depth or n_labels == 1 or n_samples <= self.min_samples_split:
            leaf_value = self._most_common(y)
            return Node(value=leaf_value)
        
        # Selects number of features from available n_feat and only selects unique features which is ensured by replace=False
        feat_idxs = np.random.choice(n_feat, self.num_features, replace=False)

        # find best split
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        left_idxs, right_idxs= self._split(X[:, best_feature], best_threshold)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_threshold, left, right)

    def _most_common(self, y):
        return Counter(y).most_common(1)[0][0]


    def _best_split(self, X, y, feat_idxs):
        # IG = E(parent) - (weighted mean)*E(children)
        # Check entropy for for some features
        # E(x) = -sigma(p(X)*log(p(X)))
        # p(X) = #x/n
        
        # Checking parent entropy (parent over here is X, y sets)

        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)

            for threshold in thresholds:        
                # Entropy of each threshold
                gain = self._info_gained(y, X_col, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _info_gained(self, y, X_col, threshold):
        # get parent entropy
        parent_entropy = self._entropy(y)

        # calculate weighted entropy of children

        left_idxs, right_idxs = self._split(X_col, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_r = len(right_idxs)
        n_l = len(left_idxs)

        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])        

        child_entropy = ((n_l/n)*e_l) + ((n_r/n)*e_r)

        # Calculate information gained
        info_gained = parent_entropy - child_entropy
        return info_gained


    def _entropy(self, y):
        # Counts occurances of each value in the array
        hist = np.bincount(y)

        # Basically contains an array of probabilities
        ps = hist/len(y)

        # Each probability
        return -np.sum([p*np.log2(p) for p in ps if p > 0])
    
    def _split(self, X_col, split_threshold):
        left_idxs = np.argwhere(X_col <= split_threshold).flatten()
        right_idxs = np.argwhere(X_col > split_threshold).flatten()

        return left_idxs, right_idxs
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X_test):
        return np.array([self._traverse_tree(x, self.root) for x in X_test])
