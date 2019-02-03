import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--points',
        metavar='PTS', required=True,
        help='List of points to plot'
    )
    parser.add_argument(
        '-sz', '--size',
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

    xs, ys = parse_points(args.points)
    f, ax = plt.subplots(1, 1, figsize=[int(x) for x in args.canvas_size.split('x')])
    if args.hide_axes:
        ax.set_axis_off()
    if args.show_grid:
        ax.grid(True)
    ax.plot(xs, ys)
    fmt = args.image_format
    f.tight_layout()
    f.savefig(f'{args.out}.{fmt}', format=fmt)


def parse_points(values):
    xs, ys = [], []
    for point in values.split(';'):
        x, y = point.split(',')
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys



if __name__ == '__main__':
    main()
