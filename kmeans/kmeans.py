"""
K-Means implementation using Numpy library.

The algorithm's pseudo-code used to create this solution was taken from:

    http://stanford.edu/~cpiech/cs221/handouts/kmeans.html

"""
import csv
import numpy as np


def kmeans(X, n_clusters=5, n_restarts=10, max_iterations=300):
    """
    Applies K-Means clustering algorithm to numerical dataset represented as
    2-dimensional NumPy array.

    Args:
        X (n_samples, n_features): 2D-array with dataset.
        n_clusters: Number of centroid vectors.
        n_restarts: Number of algorithm restarts before final solution is
            returned. The algorithm is non-deterministic as soon as its result
            depends on centroids initialization.
        max_iterations: Maximum number of iterations per algorithm restart.

    Returns:
        centroids (num_of_clusters, n_features): The 2D-array with centroids

    """
    def converged(old, new, current_iter):
        return np.allclose(old, new) or current_iter >= max_iterations

    X = normalize_dataset(X)
    n_features = X.shape[1]

    best_centroids = None
    best_score = np.inf
    labels = None

    for i in range(n_restarts):
        centroids = generate_random_centroids(n_features, n_clusters)
        old_centroids = np.zeros_like(centroids)
        count = 0

        while not converged(old_centroids, centroids, count):
            old_centroids = centroids
            labels = assign_labels(X, centroids)
            centroids = calculate_centroids(X, labels, n_clusters)
            count += 1

        score = inertia(X, labels, centroids)
        if score < best_score:
            best_centroids = centroids
            best_score = score

    return best_centroids, best_score


def normalize_dataset(X):
    """Re-scales dataset to mean=0 and std=1."""

    X -= np.mean(X, axis=0)
    X /= np.clip(np.std(X, axis=0), 10e-6, np.inf)
    return X


def generate_random_centroids(n_features, n_clusters):
    """Generates random centroids vectors."""

    return np.random.normal(size=(n_clusters, n_features))


def assign_labels(X, centroids):
    """
    Assigns cluster label to each dataset sample based on distances from this
    sample to each of centroids.
    """
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        distance_to_centroid = np.linalg.norm(X - centroid, axis=1)
        distances[:, i] = distance_to_centroid
    labels = distances.argmin(axis=1)
    return labels


def calculate_centroids(X, labels, n_clusters):
    """
    Calculates centroids based on cluster labels assigned to dataset labels.

    Each centroid is calculated as an average of vectors, belonging to the same
    cluster. In case if cluster doesn't have any samples assigned, its centroid
    is randomly re-initialized.

    Args:
        X (n_samples, n_features): Clustered dataset.
        labels: Clusters currently assigned to dataset samples.

    Returns:
        centroids (n_clusters, n_features): The new centroids array.

    """
    n_features = X.shape[1]
    centroids = np.zeros((n_clusters, n_features))
    counts = np.zeros(n_clusters)

    for sample, label in zip(X, labels):
        centroids[label, :] += sample
        counts[label] += 1

    for i in range(n_clusters):
        n = counts[i]
        if n == 0:
            # re-initialize if cluster is empty
            centroids[i, :] = np.random.normal(size=n_features)
        else:
            centroids[i, :] /= n

    return centroids


def inertia(X, labels, centroids):
    """Computes clustering inertia score which is measured as total sum of
    distances from samples to their cluster's centers

    Args:
        X (n_samples, n_features): Clustered dataset.
        labels: Clusters labels assignment.
        centroids: Clusters centroids.

    Returns:
        total_inertia: The clusters assignment inertia.

    """
    if labels is None or centroids is None:
        return np.inf

    total_inertia = 0.0
    for i, centroids in enumerate(centroids):
        [indexes] = np.where(labels == i)
        distances = np.linalg.norm(X[indexes] - centroids, axis=1)
        cluster_inertia = np.sum(distances)
        total_inertia += cluster_inertia
    return total_inertia


def read_csv(filename):
    with open(filename) as fp:
        reader = csv.reader(fp)
        _ = next(reader)  # skip header
        X, y = [], []
        for line in reader:
            *features, label = line
            X.append(features)
            y.append(label)
    return np.array(X, dtype=np.float), np.array(y, dtype=np.float)
