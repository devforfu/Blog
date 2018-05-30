from os.path import join

import numpy as np

from clustering import kmeans
from dataset import read_files


def quantize(dataset_path, k):
    """
    Applies K-Means clustering to dataset of files with accelerometer tracking
    data and returns a matrix of feature vectors created from centroids
    concatenated together.
    """
    dataset = []

    # don't need category to apply clustering
    for i, (points, _) in enumerate(read_files(dataset_path), 1):
        print('Sample %03d | number of observations: %d' % (i, len(points)))
        dataset.append(quantize_single_sample(points, k))

    return np.array(dataset)


def quantize_single_sample(points, k):
    """
    Applies quantization to a single sample with accelerometer observations.
    """
    X = np.asarray(points, dtype=np.float)
    centroids, _ = kmeans(X, n_clusters=k)
    feature_vector = centroids.flatten()
    return feature_vector


def main():
    n_clusters = 5
    dataset_path = join('datasets', 'adl')

    print('Parsing dataset:', dataset_path)
    print('Applying K-Means clustering with k=%d' % n_clusters)

    dataset = quantize(dataset_path, n_clusters)

    print('Quantized dataset shape:', dataset.shape)


if __name__ == '__main__':
    main()
