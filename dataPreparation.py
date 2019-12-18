from typing import Generator, Tuple
import cv2
import numpy as np
import os
import math

from numpy.fft import fft2, fftshift
from keras_preprocessing.image.numpy_array_iterator import NumpyArrayIterator


# FIXME: this function is a temporary _hack_
# should be replaced when refactoring to use
# generators everywhere
def to_generator(
    X: np.array, y: np.array, batch_size: int
) -> Generator[Tuple[np.array, np.array], None, None]:
    """
    :param X:          (example_nb, image_width, image_height, colors_nb)
    :param y:          (example_nb, classes_nb)
    :param batch_size: 
    :yield:
    """
    while True:
        batch_nb = math.floor(X.shape[0] / batch_size)
        for i in range(batch_nb):
            cur_idx = i * batch_size
            yield X[cur_idx : cur_idx + batch_size], y[cur_idx : cur_idx + batch_size]


def fft_process(
    batch_generator: Generator,
) -> Generator[Tuple[np.array, np.array], None, None]:
    """
    :param batch_generator: 
    :yield: tuple of np.array
        new_X : (batch_size, image_width, image_height, 2)
        y :     (batch_size, classes_nb)
    """
    while True:
        for X, y in batch_generator:
            batch_size = X.shape[0]
            image_width = X.shape[1]
            image_height = X.shape[2]
            new_X = np.zeros((batch_size, image_width, image_height, 2))
            # for each image in batch
            for i in range(batch_size):
                hann = np.mean(X[i], -1) * np.hanning(image_width)
                tmp_fft = fftshift(fft2(hann))
                new_X[i][:, :, 0] = np.log(np.abs(tmp_fft) ** 2)
                new_X[i][:, :, 1] = np.arctan(tmp_fft.imag / tmp_fft.real)
            yield new_X, y


def preprocess(image, imageWidth, imageHeight):
    return cv2.resize(image, (imageWidth, imageHeight))


def loadData(path, augment=False, preprocess=None):
    x = None
    y = np.empty(shape=(0,))
    subDirs = [
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    ]

    for classNumber, subDir in enumerate(subDirs):
        tempX = []
        tempY = []
        classDirPath = os.path.join(path, subDir)
        imageDir = [
            name
            for name in os.listdir(classDirPath)
            if os.path.isfile(os.path.join(classDirPath, name))
        ]
        for index, imageName in enumerate(imageDir):
            image = cv2.imread(os.path.join(classDirPath, imageName))
            if preprocess:
                image = preprocess[0](image, preprocess[1], preprocess[2])

            tempX.insert(index, image)
            tempY.insert(index, subDir)
        #             break;
        if x is None:
            x = np.array(tempX)
        else:
            x = np.append(x, np.array(tempX), axis=0)
        y = np.append(y, np.array(tempY), axis=0)
    return x, y
