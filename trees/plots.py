import random
from io import StringIO
from string import digits

import graphviz
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz


class Node:

    def __init__(self, feature, value, classes, left=None, right=None):
        self.feature = feature
        self.value = value
        self.classes = classes
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


def node_id(size=20, chars=digits):
    """Generates a random node ID for Graphviz file."""
    return ''.join([random.choice(chars) for _ in range(size)])


def create_graph(tree, output_file=None, rounded=True,
                 filled=True, leaves_parallel=True, rotate=True):

    styling = {
        'shape': 'box',
        'style': 'filled, rounded',
        'color': 'white'}

    # n_colors = len(np.unique(tree.classes))
    # colors = _color_brew(n_colors)

    def recurse(node, file):
        uid = node_id()
        file.write('%s [label="feature %d"];\n' % (uid, node.feature))
        edge = '%s -> %s [labeldistance=2.5, labelangle=45];\n'

        if node.left is not None:
            l_uid = recurse(node.left, file)
            file.write(edge % (uid, l_uid))

        if node.right is not None:
            r_uid = recurse(node.right, file)
            file.write(edge % (uid, r_uid))

        return uid


    opened_file = False
    try:
        if output_file is None:
            fp = StringIO()
        else:
            fp = open(output_file, 'w', encoding='utf-8')
            opened_file = True
        styles = ['%s="%s"' % (k, v) for k, v in styling.items()]
        fp.write('digraph Tree {\n')
        fp.write('node [%s];\n' % ', '.join(styles))
        recurse(tree, fp)
        fp.write('}')
        return fp.getvalue()

    finally:
        if opened_file:
            fp.close()


def get_color():
    pass


def _color_brew(n):
    """Generate n colors with equally spaced hues.
    Parameters
    ----------
    n : int
        The number of colors required.
    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
               (int(255 * (g + m))),
               (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


def main():
    tree = Node(
        feature=3, value=25.4,
        left=Node(feature=1, value=1.97),
        right=Node(
            feature=2, value=21.24,
            left=Node(
                feature=8, value=5.6,
                left=Node(feature=11, value=9.5)),
            right=Node(feature=5, value=7.39)))

    dot_data = create_graph(tree)
    graph = graphviz.Source(dot_data)
    graph.render('tree')


if __name__ == '__main__':
    main()
