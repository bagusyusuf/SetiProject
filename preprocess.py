from typing import Generator, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
from keras_preprocessing.image.directory_iterator import DirectoryIterator


def preprocess(
    image_iterator: DirectoryIterator,
) -> Generator[Tuple[np.array, np.array], None, None]:
    # X shape : (batch_size, image_size, image_size, 3)
    # y shape : (batch_size, classes_nb)
    for X, y in image_iterator:
        # for each image in batch
        for i in range(X.shape[0]):
            grey_img = np.mean(X[i], -1)
            plt.subplot(121)
            plt.imshow(grey_img)
            plt.subplot(122)
            hann = grey_img * np.hanning(grey_img.shape[0])
            f_transformed = fft2(grey_img)
            plt.imshow(np.abs(fftshift(f_transformed)))
            plt.show()
        yield X, y
    return
