import re
from os import listdir
from os.path import exists, join, basename, isdir

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans


class AccelerometerDatasetReader:

    FILE_REGEX = re.compile('^Accelerometer-([\d-]+)-(\w+)-(\w\d+).txt$')

    def __init__(self, ignore_model=True):
        self.ignore_model = ignore_model
        self.samples_ = None
        self.targets_ = None
        self.encoder_ = None

    def read(self, root):
        if not exists(root):
            raise ValueError(f'path does not exist: {root}')

        samples, targets = [], []

        for folder in listdir(root):
            path = join(root, folder)
            if not isdir(path):
                continue

            is_model = folder.lower().endswith('model')
            if self.ignore_model and is_model:
                continue

            for filename in listdir(path):
                match = self.FILE_REGEX.match(basename(filename))
                if match is None:
                    continue

                filepath = join(path, filename)
                _, category, _ = match.groups()
                with open(filepath) as lines:
                    points = [[
                        int(value) for value in line.split()]
                        for line in lines]

                samples.append(points)
                targets.append(category)

        encoder = LabelEncoder()
        self.samples_ = samples
        self.targets_ = encoder.fit_transform(targets)
        self.encoder_ = encoder

    @property
    def dataset(self):
        return self.samples_, self.targets_


def main():
    root = join('datasets', 'adl')
    reader = AccelerometerDatasetReader()
    reader.read(root)
    X, y = reader.dataset
    assert len(X) == len(y)


if __name__ == '__main__':
    main()
