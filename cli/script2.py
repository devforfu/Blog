import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        dest='points', metavar='PTS', required=True,
        help='List of points to plot'
    )
    parser.add_argument(
        '-sz',
        dest='canvas_size', metavar='SZ', default='8x6',
        help='Canvas size (default: %(default)s)'
    )
    parser.add_argument(
        '-o', '--out',
        default='output.png',
        help='Path to the output image file (default: %(default)s)'
    )
    args = parser.parse_args()
    xs, ys = parse_points(args.points)
    f, ax = plt.subplots(1, 1, figsize=[int(x) for x in args.canvas_size.split('x')])
    ax.scatter(xs, ys)
    f.savefig(args.out, format='png')


def parse_points(values):
    xs, ys = [], []
    for point in values.split(';'):
        x, y = point.split(',')
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys



if __name__ == '__main__':
    main()
