"""plotter.py generates simple plots configured from standard input or JSON.

A user is required to provide list of points to be plotted and canvas
parameters to define image rendering style properties.

examples:
    $ python plotter.py stdin -p '1,2;2,3'
    $ python plotter.py stdin -p '1,1;2,2;3,3' -o plot -f svg --show-grid
    $ python plotter.py json -j path/to/the/config.json
"""
from argparse import ArgumentParser, ArgumentError, ArgumentTypeError
from argparse import RawTextHelpFormatter
import json
from os.path import exists
import sys

import matplotlib.pyplot as plt


def main():
    parser = create_parser()
    args = parser.parse_args()
    params = {
        'stdin': vars(args),
        'json': getattr(args, 'config', None)
    }[args.command]
    f, ax = plt.subplots(1, 1, figsize=params['canvas_size'])
    if params['hide_axes']:
        ax.set_axis_off()
    if params['show_grid']:
        ax.grid(True)
    ax.plot(*params['points'])
    fmt = params['image_format']
    f.tight_layout()
    f.savefig(f'{params["out"]}.{fmt}', format=fmt)


def create_parser():
    parser = CustomParser(description=__doc__,
                          formatter_class=RawTextHelpFormatter,
                          add_help=False)

    # -----------------
    # common parameters
    # -----------------

    parser.add_argument(
        '-f', '--format',
        dest='image_format', metavar='FMT', default='png',
        choices=['png', 'svg', 'pdf'],
        help='Output image format [default: %(default)s]\n'
             'Choices: {%(choices)s}\n')

    parser.add_argument(
        '-o', '--out',
        default='output',
        help='Path to the output image file [default: %(default)s]')

    commands = parser.add_subparsers(dest='sub-command')
    commands.required = True

    # -------------------------------
    # stdio-based plotting parameters
    # -------------------------------

    stdin_cmd = commands.add_parser('stdin', add_help=False)
    stdin_cmd.set_defaults(command='stdin')
    stdin_cmd.add_argument(
        '-sz', '--size',
        type=canvas_dimensions,
        dest='canvas_size', metavar='SZ', default='8x6',
        help='Canvas size (default: %(default)s)'
    )
    stdin_cmd.add_argument(
        '-p', '--points',
        type=list_of_points,
        metavar='PTS', required=True,
        help='List of points to plot'
    )
    stdin_cmd.add_argument(
        '--hide-axes',
        default=False, action='store_true',
        help='Suppress axes drawing functionality'
    )
    stdin_cmd.add_argument(
        '--show-grid',
        default=False, action='store_true',
        help='Show grid on plot'
    )

    # ------------------------------
    # json-based plotting parameters
    # ------------------------------

    json_cmd = commands.add_parser('json', add_help=False)
    json_cmd.set_defaults(command='json')
    json_cmd.add_argument(
        '-j', '--json',
        required=True,
        metavar='PATH', dest='config', type=json_file,
        help='Path to json file'
    )

    return parser


class CustomParser(ArgumentParser):

    def error(self, message):
        self.print_help()
        sys.exit(1)


def list_of_points(value: str) -> tuple:
    """Ensures that `points` argument contains a valid sequence of values."""

    try:
        points = value.split(';')
        xs, ys = [], []
        for point in points:
            x, y = point.split(',')
            xs.append(float(x))
            ys.append(float(y))
        return xs, ys
    except ValueError:
        raise ArgumentTypeError('should have format: 1,2;2,3;3,4')


def canvas_dimensions(value: str) -> tuple:
    """Ensures that `canvas_size` represents a valid (w, h) tuple."""

    try:
        w, h = [int(v) for v in value.split('x')]
        return w, h
    except (ValueError, TypeError):
        raise ArgumentTypeError('should have format: 3x4')


def json_file(value: str) -> str:
    """Ensures that file exists and contains valid JSON object."""

    if not exists(value):
        raise ArgumentError(value, 'file doesn\'t exist')
    try:
        with open(value) as file:
            content = json.load(file)
        return content
    except json.JSONDecodeError:
        raise ArgumentTypeError('invalid JSON file')


if __name__ == '__main__':
    main()
