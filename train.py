import argparse

from config import Defaults

import numpy as np
from resNetV1 import resnet_v1
from learningRateScheduler import lr_schedule
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    ReduceLROnPlateau,
)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "-iz",
        "--image-size",
        type=int,
        default=Defaults.image_size,
        help="Image size (image will be image_size * image_size)",
    )
    arg_parser.add_argument(
        "-bz", "--batch-size", type=int, default=Defaults.batch_size, help="Batch size",
    )
    arg_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=Defaults.epochs,
        help="Number of training epochs",
    )

    args = arg_parser.parse_args()

    trainDatagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=True,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.0,
        # set range for random zoom
        zoom_range=0.0,
        # set range for random channel shifts
        channel_shift_range=0.0,
        # set mode for filling points outside the input boundaries
        fill_mode="nearest",
        # value used for fill_mode = "constant"
        cval=0.0,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        # rescale=None,
        rescale=1.0 / 255,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0,
    )

    testDatagen = ImageDataGenerator(
        samplewise_std_normalization=True, rescale=1.0 / 255
    )

    train_it = trainDatagen.flow_from_directory(
        "./seti-data/primary_small/train",
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    valid_it = testDatagen.flow_from_directory(
        "./seti-data/primary_small/valid",
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    test_it = testDatagen.flow_from_directory(
        "./seti-data/primary_small/test",
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )

    # resnet V1
    model = resnet_v1(
        input_shape=(args.image_size, args.image_size, 3),
        depth=3 * 6 + 2,
        num_classes=len(train_it.class_indices),
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=lr_schedule(0)),
        metrics=["accuracy"],
    )

    model.summary()

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(
        factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6
    )
    checkpointer = ModelCheckpoint(
        filepath="./modelCheckpoin/weights.h5", verbose=1, save_best_only=True
    )
    callbacks = [lr_reducer, lr_scheduler, checkpointer]
    model.fit_generator(
        train_it, epochs=args.epochs, validation_data=valid_it, callbacks=callbacks
    )

    scores = model.evaluate_generator(test_it, verbose=0)
    print(scores)
    model.save("./model/my_model_resV1.h5")

