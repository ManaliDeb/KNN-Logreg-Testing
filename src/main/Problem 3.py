import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # distances between x and all examples
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # k nearest neighbor indices
        k_indices = np.argsort(distances)[:self.k]
        # labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]