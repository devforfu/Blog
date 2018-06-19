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
    n_classes = len(classes)
    n_train = int(n * train_size)

    sample = np.random.choice(
        classes, size=n_train,
        replace=True, p=probs).astype(int)

    samples_per_class = np.zeros(n_classes, dtype=int)

    for value in sample:
        samples_per_class[value] += 1

    for i in range(n_classes):
        samples_per_class[i] = max(samples_per_class[i], 1)

    index = np.arange(n)
    train_samples = []
    for label, n_samples in enumerate(samples_per_class):
        subset = np.random.choice(
            index[y == label], size=n_samples, replace=False)
        train_samples.extend(subset)

    train_samples = np.array(train_samples)
    test_samples = np.array([i for i in index if i not in train_samples])
    return X[train_samples], X[test_samples], y[train_samples], y[test_samples]


def bootstrapped_sample(size):
    """Returns an index to take a sample (with replacement) of original dataset
    to create a new bootstrapped dataset of the same size.
    """
    return np.random.choice(size, size=size, replace=True)


def encode_labels(y):
    """Converts categorical targets into numerical labels."""

    unique_values = sorted(set(y))
    encoder = {key: i for i, key in enumerate(unique_values)}
    labels = np.array([encoder[target] for target in y])
    return labels, encoder, unique_values


class LoggingMixin:
    """
    Adds a set of logging methods to a class.

    The class instances should have the "log" attribute which should be a
    valid logger instance. Otherwise, logging statements are ignored.
    """

    def debug(self, msg, *args):
        self.send_to_log('debug', msg, *args)

    def info(self, msg, *args):
        self.send_to_log('info', msg, *args)

    def warning(self, msg, *args):
        self.send_to_log('warning', msg, *args)

    def error(self, msg, *args):
        self.send_to_log('error', msg, *args)

    def send_to_log(self, level, msg, *args):
        log = hasattr(self, 'log') and getattr(self, 'log', None)
        if log is None:
            return
        method = getattr(log, level)
        method(msg, *args)
