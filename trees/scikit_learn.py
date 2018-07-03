import re
from os import listdir
from os.path import exists, join, basename, isdir

import numpy as np

from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
    def verbose_classes(self):
        classes = [
            class_name.replace('_', ' ').title() 
            for class_name in self.encoder_.classes_]
        return classes

    @property
    def dataset(self):
        return self.samples_, self.targets_


class BatchTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, base_transformer):
        self.base_transformer = base_transformer

    def fit(self, X, y=None):
        return self

    def transform(self, batch, y=None):
        if y is not None:
            raise ValueError(
                'cannot apply batch transformer in supervised fashion')

        transformed = []
        for record in batch:
            transformer = clone(self.base_transformer)
            transformed.append(transformer.fit_transform(record))

        return transformed


class KMeansQuantization(BaseEstimator, TransformerMixin):

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        for record in X:
            kmeans = KMeans(n_clusters=self.k)
            kmeans.fit_transform(record)
            feature_vector = kmeans.cluster_centers_.flatten()
            features.append(feature_vector)

        return np.array(features)
    

def main():
    root = join('datasets', 'adl')
    reader = AccelerometerDatasetReader()
    reader.read(root)
    X, y = reader.dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    pipeline = make_pipeline(
        BatchTransformer(StandardScaler()),
        KMeansQuantization(k=3),
        RandomForestClassifier(n_estimators=1000))
    pipeline.fit(X_train, y_train)
    y_preds = pipeline.predict(X_test)

    acc = np.mean(y_preds == y_test)
    print(f'Dataset accuracy: {acc:2.2%}')


if __name__ == '__main__':
    main()
