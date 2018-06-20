import re
from os import listdir
from os.path import exists, join, basename, isdir

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


FILE_REGEX = re.compile('^Accelerometer-([\d-]+)-(\w+)-(\w\d+).txt$')


class AccelerometerDatasetReader(TransformerMixin):

    def __init__(self, ignore_model=True):
        self.ignore_model = ignore_model

    def fit(self, *args, **kwargs):
        return self

    def transform(self, path):
        if not exists(path):
            raise ValueError(f'path does not exist: {path}')

        records = [record for record in read_files(path, self.ignore_model)]
        samples, targets = zip(*records)
        self.samples_ = samples
        self.targets_ = targets




def read_files(root: str, ignore_model: bool=True):
    """Recursively reads accelerometer data files from root folder."""

    for folder in listdir(root):
        path = join(root, folder)
        if not isdir(path):
            continue
        is_model = folder.lower().endswith('model')
        if ignore_model and is_model:
            continue
        for filename in listdir(path):
            record = read_file(join(path, filename))
            if record is None:
                continue
            yield record


def read_file(filename: str) -> dict:
    """Parses accelerometer measurements from file."""

    match = FILE_REGEX.match(basename(filename))
    if match is None:
        return None
    _, category, _ = match.groups()
    with open(filename) as lines:
        points = [[
            int(value) for value in line.split()]
            for line in lines]
    return points, category


def get_class_names(root):
    """Returns list of dataset classes."""

    return [folder.lower().replace('_', ' ').title()
            for folder in listdir(root)
            if isdir(join(root, folder))
            and not folder.endswith('MODEL')]


if __name__ == '__main__':
    main()
