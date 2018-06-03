import csv
from collections import Counter
from os.path import join, dirname

import graphviz
import numpy as np

from plots import create_graph
from decision_tree import learn_tree


def read_csv(filename):
    with open(filename) as fp:
        reader = csv.reader(fp)
        header = next(reader)  # skip header
        X, y = [], []
        for line in reader:
            *features, label = line
            X.append(features)
            y.append(label)
    return header, np.array(X, dtype=np.float), np.array(y, dtype=np.float)


def train_test_split(X, y, train_size=0.8):
    assert X.shape[0] == y.shape[0]

    n = X.shape[0]
    counts = Counter(y)
    classes, probs = zip(*[(k, v/n) for k, v in counts.items()])
    n_train = int(n * train_size)

    sample = np.random.choice(classes, size=n_train, replace=True, p=probs)
    samples_per_class = np.bincount(sample)
    train_samples = []
    for label, n_samples in enumerate(samples_per_class):
        index = np.random.choice(y == label, size=n_samples, replace=False)
        train_samples.extend(index)

    train_samples = np.array(train_samples)
    test_samples = ~train_samples
    return X[train_samples], X[test_samples], y[train_samples], y[test_samples]


def encode_labels(y):
    """Converts categorical targets into numerical labels."""

    unique_values = sorted(set(y))
    encoder = {key: i for i, key in enumerate(unique_values)}
    labels = np.array([encoder[target] for target in y])
    return labels, encoder, unique_values


def main():
    data_path = join(dirname(__file__), 'datasets', 'wine.csv')
    feature_names, X, y = read_csv(data_path)
    y, encoder, class_names = encode_labels(y)
    tree = learn_tree(X, y)
    dot_data = create_graph(tree, feature_names, class_names)
    graph = graphviz.Source(dot_data)
    graph.render('tree')


if __name__ == '__main__':
    main()
