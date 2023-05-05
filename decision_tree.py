import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(len(set(y)))]
        if not num_samples_per_class:
            return Node(value=0)

        predicted_class = np.argmax(num_samples_per_class)
        node = Node(value=predicted_class)

        if depth < self.max_depth:
            if len(X) >= self.min_samples_split:
                feat_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)

                best_feat, best_thresh = self._best_split(X, y, feat_idxs)
                if best_feat is not None:
                    left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
                    left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
                    right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
                    node = Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

        return node

    def _best_split(self, X, y, feat_idxs):
        best_feat, best_thresh = None, None
        max_ig = -1

        for i in feat_idxs:
            X_column = X[:, i]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                ig = self._information_gain(y, X_column, threshold)

                if ig > max_ig:
                    max_ig = ig
                    best_feat = i
                    best_thresh = threshold

        return best_feat, best_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
