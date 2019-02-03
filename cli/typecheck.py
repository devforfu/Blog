from argparse import ArgumentParser, ArgumentTypeError
import matplotlib.pyplot as plt


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--points',
        type=list_of_points,
        metavar='PTS', required=True,
        help='List of points to plot'
    )
    parser.add_argument(
        '-sz', '--size',
        type=canvas_dimensions,
        dest='canvas_size', metavar='SZ', default='8x6',
        help='Canvas size (default: %(default)s)'
    )
    parser.add_argument(
        '-f', '--format',
        dest='image_format', metavar='FMT', default='png',
        choices=['png', 'svg', 'pdf'],
        help='Output image format (choices: %(choices)s, default: %(default)s)'
    )
    parser.add_argument(
        '-o', '--out',
        default='output',
        help='Path to the output image file (default: %(default)s)'
    )
    parser.add_argument(
        '--hide-axes',
        default=False, action='store_true',
        help='Suppress axes drawing functionality'
    )
    parser.add_argument(
        '--show-grid',
        default=False, action='store_true',
        help='Show grid on plot'
    )

    args = parser.parse_args()

    f, ax = plt.subplots(1, 1, figsize=args.canvas_size)
    if args.hide_axes:
        ax.set_axis_off()
    if args.show_grid:
        ax.grid(True)
    ax.plot(*args.points)
    fmt = args.image_format
    f.tight_layout()
    f.savefig(f'{args.out}.{fmt}', format=fmt)


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


if __name__ == '__main__':
    main()
