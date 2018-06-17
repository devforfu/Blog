import csv
from collections import Counter
from os.path import join, dirname

import graphviz
import numpy as np

from plots import create_graph
from decision_tree import learn_tree, predict_tree


PALETTE = [
    [167, 98, 228],
    [74, 144, 226],
    [244, 166, 35]
]


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

    index = np.arange(n)
    sample = np.random.choice(classes, size=n_train, replace=True, p=probs)
    samples_per_class = np.bincount(sample.astype(int))
    train_samples = []

    for label, n_samples in enumerate(samples_per_class):
        subset = np.random.choice(
            index[y == label], size=n_samples, replace=False)
        train_samples.extend(subset)

    train_samples = np.array(train_samples)
    test_samples = np.array([i for i in index if i not in train_samples])
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
    y, encoder, class_names = encode_labels(y.astype(int))

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    tree = learn_tree(X_train, y_train, max_depth=5)
    preds = predict_tree(tree, X_test)
    acc = np.mean(y_test == preds)
    print(f'Test set accuracy: {acc:2.2%}')

    dot_data = create_graph(tree, feature_names, class_names, palette=PALETTE)
    graph = graphviz.Source(dot_data, format='png')
    graph.render('tree.dot')


if __name__ == '__main__':
    main()
