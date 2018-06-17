import csv
from collections import Counter

import numpy as np


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
