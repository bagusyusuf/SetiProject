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
            plt.subplot(121)
            plt.imshow(X[i])
            plt.subplot(122)
            test = fft2(X[i])
            plt.imshow(20 * np.log(np.abs(fftshift(test))))
            plt.show()
            # fftn()
        yield X, y
    return
