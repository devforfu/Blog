import math
from os import listdir
from os.path import join

import numpy as np
from PIL import Image
from Augmentor.Operations import Rotate, Flip
from sklearn.preprocessing import LabelBinarizer


PATH_TO_IMAGES = '/Users/ck/Code/Tasks/blog/generators/images'


def dataset(root_folder, batch_size=32):
    """
    Source generator which parses folders with training samples and preparing
    label encoder to convert each image's class into one-hot encoded vector.

    The generator yields file names and encoded labels in batches of size equal
    to `batch_size` parameter value.

    Should be the very first generator in pipeline providing data for
    subsequent steps.
    """
    images_and_classes = []
    for image_class in listdir(root_folder):
        subfolder = join(root_folder, image_class)
        for sample in listdir(subfolder):
            filename = join(subfolder, sample)
            images_and_classes.append((filename, image_class))

    n_batches = int(math.ceil(len(images_and_classes) / batch_size))
    classes = [c for (img, c) in images_and_classes]
    binarizer = LabelBinarizer()
    binarizer.fit(classes)

    start = 0
    for _ in range(n_batches):
        batch = images_and_classes[start:(start + batch_size)]
        paths, labels = zip(*batch)
        encoded = binarizer.transform(labels)
        start += batch_size
        yield np.asarray(paths), encoded


def read_images(target_size=(224, 224)):
    """
    Reads images from disk and rescales them to `target_size`.
    """
    while True:
        filenames, y = yield
        images = []
        for sample in filenames:
            img = Image.open(sample)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(target_size, Image.NEAREST)
            images.append(img)
        yield images, y


def augment(horizontal_flip=True,
            vertical_flip=False,
            rotate90=False,
            probability=0.5):
    """
    Applies a group of augmentation operations to each sample in batch.
    """

    ops = []

    if horizontal_flip:
        ops.append(Flip(
            probability=probability,
            top_bottom_left_right='LEFT_RIGHT'))

    if vertical_flip:
        ops.append(Flip(
            probability=probability,
            top_bottom_left_right='TOP_BOTTOM'))

    if rotate90:
        ops.append(Rotate(probability=probability, rotation=90))

    while True:
        images, y = yield
        for op in ops:
            images = op.perform_operation(images)
        yield images, y


def rescale_images(mean):
    """
    Subtracts mean pixel value from each channel,
    """
    assert len(mean) == 3, 'Mean should be an array of 3 elements'

    while True:
        images, y = yield
        x = np.asarray([np.asarray(img, dtype=float) for img in images])
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        x /= 256.0
        yield x, y


def shuffle_samples():
    """
    Shuffles batch samples.
    """
    while True:
        x, y = yield
        index = np.random.permutation(len(x))
        yield x[index], y[index]


class GeneratorPipeline:
    """Convenience wrapper combining a list of generators together into a
    single generator.
    """

    def __init__(self, source, *steps):
        self.source = source
        self.steps = list(steps)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        batch = next(self.source)
        self.send_none()
        transformed = self.send(batch)
        return transformed

    def send_none(self):
        for step in self.steps:
            step.send(None)

    def send(self, batch):
        x = batch
        for generator in self.steps:
            x = generator.send(x)
        return x


def main():
    pipeline = GeneratorPipeline(
        dataset(PATH_TO_IMAGES),
        read_images(),
        augment(rotate90=True),
        rescale_images(mean=[103.939, 116.779, 123.68]),
        shuffle_samples())

    for i, (x, y) in enumerate(pipeline):
        print('Batch', i, x.shape, y.shape)


if __name__ == '__main__':
    main()
