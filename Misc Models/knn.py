import numpy as np
from collections import Counter

def distance(x1, x2):
    # Takes in a two points and returns distance between them
    return np.sqrt(np.sum((x1 -x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
        
    def predict(self, test_x):
        # Predictions is an array containing all the predictions made by the model
        # the helper function _predict is called for each value of x in the testing dataset
        predictions = [self._predict(x) for x in test_x]
        return predictions

    def _predict(self, x):
        # Helper function for the actual predictor
        # Calculation of distance for a single point can be done here
        
        # Calc dist of given point w/ all points in training set
        # then get k closest values
        # compute label/average based on majority vote

        distance_arr = [distance(x, training_point) for training_point in self.x_train]

        # Sorts the distances array in ascending order and then stores the values from the start until the kth values
        # Argsort tells where the original index from distance_arr would be after sorting
        # Returns the indeces of the closes neighbours
        k_indeces = np.argsort(distance_arr)[:self.k]

        # Stores the labels from the y training set
        # Only stores the labels of those indeces that are present in the k_indeces array
        k_nearest_labels = [self.y_train[i] for i in k_indeces]

        # Counts all the labels and returns most common, might be good to read up on some documentation of this datastructure
        majority = Counter(k_nearest_labels).most_common()
        return int(majority[0][0])