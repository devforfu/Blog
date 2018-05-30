"""
Parsing utilities for ADL Recognition with Wrist-worn Accelerometer Data Set.
"""
import re
from os import listdir
from os.path import join, basename, isdir


FILE_REGEX = re.compile('^Accelerometer-([\d-]+)-(\w+)-(\w\d+).txt$')


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
