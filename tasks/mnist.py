from urllib.request import urlretrieve

import gzip

import os

import numpy as np

from tasks import data_fnm

from .helpers import labels_to_one_hot

data_fnm = os.path.join(data_fnm, 'mnist')


def load_data():
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, os.path.join(data_fnm, filename))

    def load_mnist_images(filename):
        path = os.path.join(data_fnm, filename)
        if not os.path.exists(path):
            download(filename)
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 784)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        path = os.path.join(data_fnm, filename)
        if not os.path.exists(path):
            download(filename)
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels_to_one_hot(data, 10)

    x_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return {'train': {'images': x_train,
                      'labels': y_train},
            'test': {'images': x_test,
                     'labels': y_test}}
