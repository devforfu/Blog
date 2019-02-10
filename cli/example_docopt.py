"""
Scatter plots generator.

Usage:
    scatter <filename> <X> <Y> (<X> <Y>)...

"""
from itertools import chain
import docopt
import matplotlib.pyplot as plt
from matplotlib import rcParams


default_style = {
    'font.family': 'monospace',
    'font.size': 18,
    'figure.figsize': (8, 6)
}


class Plotter:
    def __init__(self, style: dict=None):
        self.style = style or default_style
        plt.rcParams.update(self.style)

    def scatter(self, name: str, *points):
        n = len(points)
        if n % 2 != 0:
            n -= 1
        points = points[:n]
        xs, ys = points[::2], points[1::2]
        f, ax = plt.subplots(1, 1)
        ax.scatter(xs, ys)
        self.save(f, name)

    def save(self, figure: plt.Figure, output: str):
        figure.savefig(f'{output}.png', format='png')


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    pairs = [(int(x), int(y)) for x, y in zip(args['<X>'], args['<Y>'])]
    points = list(chain.from_iterable(pairs))
    filename = args['<filename>']
    plotter = Plotter()
    plotter.scatter(filename, *points)
