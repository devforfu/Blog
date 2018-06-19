import math

import numpy as np

from utils import LoggingMixin
from utils import bootstrapped_sample


class RandomForestClassifier(LoggingMixin):
    """
    A convenience wrapper on top of decision tree learning function.

    Builds an ensemble of trees and provides methods to make predictions.
    Methods signatures and computed attributes names follow scikit-learn
    naming convention.

    Arguments:

        tree_funcs (build_fn, predict_fn):
            Two functions that train a single decision tree on
            provided dataset and make predictions on it. Extra arguments
            could be passed to specify maximal depth, split sizes,
            considered features, etc.

        n_trees:
            Number of trees in ensemble.

        feature_subset_size {int, str, None}:
            Size of features subset to be considered on each tree split.
            Should be an integer, None (if all attributes should be taken
            into account), or 'sqrt' to take a square root of total number of
            dataset features.

        max_depth:
            Maximum depth of a single tree.

        min_split_size:
            Minimum number of observations in a node to split the node
            into two new nodes.

        min_leaf_size:
            Minimum number of observations in decision tree leafs.

        log:
            An instance of logging class.

    """
    def __init__(self, tree_funcs, n_trees: int=10,
                 feature_subset_size: str='sqrt', max_depth: int=5,
                 min_split_size: int=10, min_leaf_size: int=None,
                 log=None):

        if n_trees < 1:
            raise ValueError(f'cannot build an ensemble of {n_trees:d} trees')

        self.build_fn, self.predict_fn = tree_funcs
        self.n_trees = n_trees
        self.feature_subset_size = feature_subset_size
        self.max_depth = max_depth
        self.min_split_size = min_split_size
        self.min_leaf_size = min_leaf_size
        self.log = log
        self.feature_subset_size_ = None
        self.ensemble_ = None
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        m = _validate_subset_size(X, self.feature_subset_size)
        n = self.n_trees

        self.info(f'Started building an ensemble of {n} decision trees')
        self.info(f'Training dataset shape: {X.shape}')
        self.info(f'Maximal tree depth: {self.max_depth}')
        self.info(f'Minimal number of samples per node '
                  f'to make a split: {self.min_split_size}')
        self.info(f'Minimal number of samples '
                  f'to create a leaf: {self.min_leaf_size}')
        self.info(f'Number of random features considered '
                  f'per each tree split: {m}')

        string_length = len(str(n))
        ensemble = []

        for i in range(1, n + 1):
            self.debug(f'Building tree %{string_length}d of %d', i, n)
            index = bootstrapped_sample(X.shape[0])
            tree = self.build_fn(
                X=X[index], y=y[index],
                max_depth=self.max_depth,
                min_split_size=self.min_split_size,
                min_leaf_size=self.min_leaf_size,
                features_subset_size=m)
            ensemble.append(tree)

        classes = np.unique(y)
        classes.sort()

        self.ensemble_ = ensemble
        self.classes_ = classes
        self.n_classes_ = len(self.classes_)
        return self

    def predict_decisions(self, X, n_trees=None):
        """
        Returns matrix with predicted classes for each 
        instance for each of trees in ensemble.
        """
        if self.ensemble_ is None:
            raise RuntimeError('fit method should be called first')
            
        if n_trees is None:
            n_trees = self.n_trees
        elif n_trees > self.n_trees:
            n_trees = self.n_trees    
            
        predictions = np.zeros((X.shape[0], n_trees), dtype=int)
        for tree_index, tree in enumerate(self.ensemble_[:n_trees]):
            predictions[:, tree_index] = self.predict_fn(tree, X)
        
        return predictions
        
    def predict_proba(self, X, **params):
        """
        Returns matrix with probabilities per instance per class.
        """
        predictions = self.predict_decisions(X, **params)
        probabilities = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for sample_index in range(X.shape[0]):
            preds = predictions[sample_index, :]
            counts = np.zeros(self.n_classes_)
            for value in preds:
                counts[value] += 1
            ratios = counts / counts.sum()
            probabilities[sample_index, :] = ratios

        return probabilities

    def predict(self, X, **params):
        """
        Returns a vector with class predictions.
        """
        probabilities = self.predict_proba(X, **params)
        labels = probabilities.argmax(axis=1)
        return labels


def _validate_subset_size(X, size):
    """
    Checks if features subset size is equal to one of valid values.
    """
    if size is None:
        return X.shape[1]

    if isinstance(size, int) and size > X.shape[1]:
        raise ValueError(
            f'the dataset has only {X.shape[1]:d} features, '
            f'but feature subset size is equal to {size:d}')

    if size == 'sqrt':
        return int(math.sqrt(X.shape[1]))

    if not isinstance(size, int):
        raise TypeError(f'unexpected value for feature subset size: {size}')

    return size
