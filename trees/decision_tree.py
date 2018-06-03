from collections import Counter

import numpy as np


class _Node:
    def __init__(self, value):
        self.value = value

    @property
    def is_leaf(self):
        return False


class Node(_Node):
    def __init__(self, feature, value, gini, counts, depth, left, right):
        super().__init__(value)
        self.feature = feature
        self.value = value
        self.gini = gini
        self.counts = counts
        self.depth = depth
        self.left = left
        self.right = right


class Leaf(_Node):

    @property
    def is_leaf(self):
        return True

    def __eq__(self, other):
        if other.is_leaf:
            return self.value == other.value
        return False


def learn_tree(X, y, max_depth=5, min_split_size=10,
               features_subset_size=None, min_leaf_size=None) -> Node:
    """Creates a decision tree based on provided dataset.

    Returns:
        Node: The root node of the created tree.

    """
    features = np.arange(X.shape[1])
    min_leaf_size = min_leaf_size or 1


    def learn(x_index, y_index, depth):
        """Recursive function called to split the node into left and right
        children. Returns majority class in case if node cannot be split or
        contains observations belonging to the same class.

        Note that the function operates with array indexes instead of arrays
        themselves. It means that the original arrays are not copied or
        modified during training process.

        Args:
             x_index: Subset of observations in the node.
             y_index: Subset of respective observation targets.
             depth: Current depth level.

        Returns:
             Node: Decision tree node.

        """
        shortcut = (
            single_class_node(y_index) or
            too_small_for_split(x_index) or
            max_depth_exceeded(depth))

        if shortcut:
            return majority_vote(y_index)

        best_gini = np.inf
        best_feature = None
        best_split = None
        best_value = None
        targets = y[y_index]
        counts = Counter(targets)
        classes = np.unique(targets)

        if features_subset_size is None:
            considered_features = features
        else:
            considered_features = np.random.choice(
                features, size=features_subset_size, replace=False)

        for feature in considered_features:
            column = X[x_index][:, feature]
            for threshold in column:
                left = mask(column <= threshold, column)
                if too_few_samples_per_child(left):
                    continue
                gini = gini_index(y_index, classes, left)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_split = left
                    best_value = threshold

        if best_split is None:
            return majority_vote(y_index)

        x_left = x_index[best_split]
        x_right = x_index[~best_split]
        y_left = y_index[best_split]
        y_right = y_index[~best_split]

        left = learn(x_left, y_left, depth + 1)
        right = learn(x_right, y_right, depth + 1)

        if left.is_leaf and right.is_leaf and left == right:
            return Leaf(left.value)

        return Node(feature=best_feature, value=best_value, counts=counts,
                    gini=best_gini, depth=depth, left=left, right=right)


    def gini_index(node_subset, classes, left):
        """
        Computes Gini index for a node split.

        Args:
             node_subset: Subset of observations in the node.
             classes: List of unique classes represented in the node.
             left: Subset of the node's observations assigned to the left child.

        Return:
            total_gini: The node split's Gini score.

        """
        total_gini = 0.0
        n_total = node_subset.shape[0]
        right = ~left

        for branch_subset in (left, right):
            group = y[node_subset][branch_subset]
            size = group.shape[0]
            if size == 0:
                continue

            keys, values = np.unique(group, return_counts=True)
            counts = dict(zip(keys, values))
            ratios = np.array([counts.get(value, 0)/size for value in classes])
            score = (ratios ** 2).sum()
            total_gini += size*(1.0 - score)/n_total

        return total_gini


    def majority_vote(index):
        """
        Returns the majority value of an array.
        """
        arr = y[index]
        value = np.argmax(np.bincount(arr.astype(int)))
        return Leaf(value)


    def single_class_node(y_index):
        """
        Returns true if all elements of sequence are equal to the same value.
        """
        return np.unique(y[y_index]).shape[0] == 1


    def too_small_for_split(index):
        """
        Checks if node is too small to being split.
        """
        return min_split_size is not None and index.shape[0] < min_split_size


    def max_depth_exceeded(depth):
        """
        Checks if maximum tree depth is reached.
        """
        return max_depth is not None and depth >= max_depth


    def too_few_samples_per_child(index):
        """
        Checks if leaf split generates enough instances in the child node.

        Note that `index` array is a boolean mask which selects a subset of
        parent nodes to group them into child node. True values select values
        of left node, while False values - for the right one.
        """
        total = len(index)
        left_node_samples = index.sum()
        right_node_samples = total - left_node_samples
        return (left_node_samples < min_leaf_size or
                right_node_samples < min_leaf_size)


    x_root_index = np.arange(X.shape[0])
    y_root_index = np.arange(y.shape[0])
    return learn(x_root_index, y_root_index, 0)


def mask(condition, arr):
    """
    Returns a mask selecting array values meeting the condition.
    """
    return np.ma.masked_where(condition, arr).mask
