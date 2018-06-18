import numpy as np

from clustering import kmeans
from dataset import read_files


def quantize(dataset_path, k):
    """
    Applies K-Means clustering to dataset of files with accelerometer tracking
    data and returns a matrix of feature vectors created from centroids
    concatenated together.
    """
    dataset, categories = [], []

    for i, (points, category) in enumerate(read_files(dataset_path), 1):
        print('Sample %03d | number of observations: %d' % (i, len(points)))
        dataset.append(quantize_single_sample(points, k))
        categories.append(category)

    return np.array(dataset), np.array(categories)


def quantize_single_sample(points, k):
    """
    Applies quantization to a single sample with accelerometer observations.
    """
    X = np.asarray(points, dtype=np.float)
    centroids, _ = kmeans(X, n_clusters=k)
    feature_vector = centroids.flatten()
    return feature_vector
