import random
from io import StringIO
from string import digits

import numpy as np


def node_id(size=20, chars=digits):
    """Generates a random node ID for Graphviz file."""
    return ''.join([random.choice(chars) for _ in range(size)])


def create_graph(tree, feature_names, class_names, output_file=None,
                 palette=None):

    styling = {
        'shape': 'box',
        'style': 'filled, rounded',
        'color': 'black'}

    if palette is None:
        n_colors = len(class_names)
        colors = _color_brew(n_colors)
    else:
        assert len(palette) == len(class_names)
        colors = palette


    def create(node, file):
        uid = node_id()
        label = node_to_str(node)
        color = get_node_color(node)

        file.write(f'{uid} [label="{label}", fillcolor="{color}"];\n')

        if not node.is_leaf:
            edge = '%s -> %s [label=%s, labeldistance=2.5, labelangle=45];\n'

            if node.left is not None:
                l_uid = create(node.left, file)
                file.write(edge % (uid, l_uid, 'yes'))

            if node.right is not None:
                r_uid = create(node.right, file)
                file.write(edge % (uid, r_uid, 'no'))

        return uid


    def node_to_str(node):
        """
        Converts decision tree node into string representation.
        """
        if node.is_leaf:
            return class_names[node.value]
        else:
            name = feature_names[node.feature]
            total = sum(node.counts.values())
            [(category, num_of_samples)] = node.counts.most_common(1)
            ratio = num_of_samples / total
            lines = [
                f'samples: {total}',
                f'gini: {node.gini:2.2f}',
                f'ratio: {ratio:2.2f}',
                f'{name} <= {node.value:2.2f}']
            return '\n'.join(lines)


    def get_node_color(node):
        if node.is_leaf:
            color = colors[node.value]
        else:
            [(cls, count)] = node.counts.most_common(1)
            total = sum(node.counts.values())
            rgb = colors[cls]
            alpha = int(255 * count / total)
            color = rgb + [alpha]

        return to_hexadecimal(color)


    opened_file = False
    try:
        if output_file is None:
            fp = StringIO()
        else:
            fp = open(output_file, 'w', encoding='utf-8')
            opened_file = True
        styles = [f'{k}="{v}"' for k, v in styling.items()]
        fp.write('digraph Tree {\n')
        fp.write('bgcolor="#00000000"\n')
        fp.write('node [%s];\n' % ', '.join(styles))
        create(tree, fp)
        fp.write('}')
        return fp.getvalue()

    finally:
        if opened_file:
            fp.close()


def to_hexadecimal(values):
    hex_codes = [str(i) for i in range(10)]
    hex_codes.extend(['a', 'b', 'c', 'd', 'e', 'f'])
    color = [hex_codes[c // 16] + hex_codes[c % 16] for c in values]
    return '#' + ''.join(color)


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
