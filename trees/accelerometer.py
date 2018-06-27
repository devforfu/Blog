from os.path import join

import numpy as np

from quantization import quantize
from decision_tree import learn_tree
from decision_tree import predict_tree
from ensemble import RandomForestClassifier
from utils import train_test_split, encode_labels


def main():
    n_clusters = 5
    dataset_path = join('datasets', 'adl')
    X, labels = quantize(dataset_path, n_clusters)
    y, encoder, classes = encode_labels(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    random_forest = RandomForestClassifier(
        tree_funcs=(learn_tree, predict_tree),
        n_trees=50, max_depth=3, min_leaf_size=5,
        min_split_size=10, feature_subset_size='sqrt')

    random_forest.fit(X_train, y_train)
    predictions = random_forest.predict(X_test)
    acc = np.mean(predictions == y_test)

    print(f'Test dataset predictions accuracy: {acc:2.2%}')


if __name__ == '__main__':
    main()
