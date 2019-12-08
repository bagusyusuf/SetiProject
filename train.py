import os
import time
import argparse
import numpy as np
import tensorflow

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

from dataPreparation import loadData, preprocess
from config import Defaults
from helper import hms_string

from resNetV1 import resnet_v1
from sklearn import preprocessing


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "-ih",
        "--image-height",
        type=int,
        default=Defaults.image_height,
        help="Image height",
    )
    arg_parser.add_argument(
        "-iw",
        "--image-width",
        type=int,
        default=Defaults.image_width,
        help="Image width",
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

    trainPath = os.path.join("./seti-data/primary_small/train")
    valPath = os.path.join("./seti-data/primary_small/valid")
    testPath = os.path.join("./seti-data/primary_small/test")

    # parameter
    print("load data")

    (xTrain, yTrainStr) = loadData(
        trainPath, preprocess=[preprocess, args.image_width, args.image_height]
    )
    (xVal, yValStr) = loadData(
        valPath, preprocess=[preprocess, args.image_width, args.image_height]
    )
    (xTest, yTestStr) = loadData(
        testPath, preprocess=[preprocess, args.image_width, args.image_height]
    )

    print("data loaded")

    # prepare dataset
    le = preprocessing.LabelEncoder()
    yTrain = le.fit_transform(yTrainStr)
    yVal = le.transform(yValStr)
    yTest = le.transform(yTestStr)

    # predifine
    NUM_CLASSES = np.unique(yTrain).shape[0]  # 7

    # Subtracting pixel mean improves accuracy
    SUBTRACT_PIXEL_MEAN = True

    COLORS = xTrain.shape[3]  # 3

    # Input image dimensions.
    input_shape = xTrain.shape[1:]

    # Normalize data.
    xTrain = xTrain.astype("float32") / 255
    xVal = xVal.astype("float32") / 255
    xTest = xTest.astype("float32") / 255

    # If subtract pixel mean is enabled
    if SUBTRACT_PIXEL_MEAN:
        xTrainNMean = np.mean(xTrain, axis=0)
        xTrain -= xTrainNMean
        xVal -= xTrainNMean
        xTest -= xTrainNMean

    # Convert class vectors to binary class matrices.
    yTrain = tensorflow.keras.utils.to_categorical(yTrain, NUM_CLASSES)
    yVal = tensorflow.keras.utils.to_categorical(yVal, NUM_CLASSES)
    yTest = tensorflow.keras.utils.to_categorical(yTest, NUM_CLASSES)

    # resnet V1
    USE_AUGMENTATION = True
    DEPTH = COLORS * 6 + 2

    model = resnet_v1(input_shape=input_shape, depth=DEPTH, num_classes=NUM_CLASSES)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=lr_schedule(0)),
        metrics=["accuracy"],
    )

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
        target_size=(args.image_width, args.image_height),
        batch_size=args.batch_size,
    )
    valid_it = testDatagen.flow_from_directory(
        "./seti-data/primary_small/valid",
        target_size=(args.image_width, args.image_height),
        batch_size=args.batch_size,
    )
    test_it = testDatagen.flow_from_directory(
        "./seti-data/primary_small/test",
        target_size=(args.image_width, args.image_height),
        batch_size=args.batch_size,
    )

    # resnet V1
    model = resnet_v1(
        input_shape=(args.image_width, args.image_height, 3),
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

    # calculate time
    start_time = time.time()

    checkpointer = ModelCheckpoint(
        filepath="./modelCheckpoin/weights.h5", verbose=1, save_best_only=True
    )
    callbacks = [lr_reducer, lr_scheduler, checkpointer]
    # model.fit_generator(train_it, epochs=EPOCHS,
    #                     validation_data=valid_it, callbacks=callbacks)

    # Run training, with or without data augmentation.
    if not USE_AUGMENTATION:
        print("Not using data augmentation.")
        model.fit(
            xTrain,
            yTrain,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(xVal, yVal),
            shuffle=True,
            callbacks=callbacks,
        )
    else:
        print("Using real-time data augmentation.")
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
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
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0,
        )

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(xTrain)

        with tensorflow.device("/gpu:0"):
            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(
                datagen.flow(xTrain, yTrain, batch_size=args.batch_size),
                validation_data=(xVal, yVal),
                epochs=args.epochs,
                verbose=1,
                workers=1,
                callbacks=callbacks,
                use_multiprocessing=False,
            )

    elapsed_time = time.time() - start_time

    # scores = model.evaluate_generator(test_it, verbose=0)
    scores = model.evaluate(xTest, yTest, verbose=0)
    print("128Elapsed time: {}".format(hms_string(elapsed_time)))
    print(scores)
    model.save("./model/test_96_resV1.h5")

