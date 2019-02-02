import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        required=True,
        help='List of points to plot'
    )
    parser.add_argument(
        '-sz',
        default='8x6',
        help='Canvas size'
    )
    parser.add_argument(
        '-o', '--out',
        default='output.png',
        help='Path to the output image file'
    )
    args = parser.parse_args()
    xs, ys = parse_points(args.p)
    f, ax = plt.subplots(1, 1, figsize=[int(x) for x in args.sz.split('x')])
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
