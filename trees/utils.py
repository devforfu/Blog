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
    from collections import Counter

    samples_per_class = Counter(y)

    index = np.arange(X.shape[0])
    train_samples = []

    for value, count in samples_per_class.items():
        n_samples = int(count * train_size)
        subset = np.random.choice(
            index[y == value], size=n_samples, replace=False)
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
