import string
from os.path import dirname, join

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .kmeans import read_csv
from .kmeans import kmeans
from .kmeans import assign_labels


DEFAULT_PALETTE = ['red', 'darkorange', 'royalblue', 'green', 'magenta', 'pink']


def palette(colors=None):
    if colors is None:
        colors = DEFAULT_PALETTE
    n = len(colors)

    def wrapper(value):
        return colors[value % n]

    return wrapper


def main():
    np.random.seed(1)

    data_path = join(dirname(__file__), 'datasets', 'blobs.csv')
    X, _ = read_csv(data_path)

    figure, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    font_path = join(dirname(__file__), 'fonts', 'RobotoSlab-Regular.ttf')
    font = font_manager.FontProperties(fname=font_path)
    font.set_size(16)
    gray = '#3c3c3c'

    for i, n_clusters in enumerate((2, 3, 4, 5)):
        print('Running K-means with k=%d' % n_clusters)
        centroids, score = kmeans(X, n_clusters)
        print('Best inertia score: %.2f' % score)

        letter = string.ascii_letters[i]
        title = '(%s) k=%d, inertia=%2.2f' % (letter, n_clusters, score)
        labels = assign_labels(X, centroids)
        get_color = palette()
        colors = [get_color(l) for l in labels]

        axes[i].scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.6)
        axes[i].set_title(title, fontproperties=font, color=gray)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        for spine in ('top', 'right', 'bottom', 'left'):
            axes[i].spines[spine].set_color(gray)

        for (x, y) in centroids:
            axes[i].plot(
                x, y, color='white',
                markeredgewidth=1, markeredgecolor=gray,
                markersize=10, marker='d')

    figure.tight_layout()
    figure.savefig('clusters.png', transparent=False)


if __name__ == '__main__':
    main()
