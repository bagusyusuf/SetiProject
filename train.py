import os
import time
import numpy as np
import tensorflow

from resNetV1 import resnet_v1
from learningRateScheduler import lr_schedule
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from dataPreparation import loadData, preprocess


if __name__ == "__main__":

    # BATCH_SIZE = 32

    # trainDatagen = ImageDataGenerator(
    #     # set input mean to 0 over the dataset
    #     featurewise_center=False,
    #     # set each sample mean to 0
    #     samplewise_center=False,
    #     # divide inputs by std of dataset
    #     featurewise_std_normalization=False,
    #     # divide each input by its std
    #     samplewise_std_normalization=True,
    #     # apply ZCA whitening
    #     zca_whitening=False,
    #     # epsilon for ZCA whitening
    #     zca_epsilon=1e-06,
    #     # randomly rotate images in the range (deg 0 to 180)
    #     rotation_range=0,
    #     # randomly shift images horizontally
    #     width_shift_range=0.1,
    #     # randomly shift images vertically
    #     height_shift_range=0.1,
    #     # set range for random shear
    #     shear_range=0.,
    #     # set range for random zoom
    #     zoom_range=0.,
    #     # set range for random channel shifts
    #     channel_shift_range=0.,
    #     # set mode for filling points outside the input boundaries
    #     fill_mode='nearest',
    #     # value used for fill_mode = "constant"
    #     cval=0.,
    #     # randomly flip images
    #     horizontal_flip=True,
    #     # randomly flip images
    #     vertical_flip=False,
    #     # set rescaling factor (applied before any other transformation)
    #     # rescale=None,
    #     rescale=1./255,
    #     # set function that will be applied on each input
    #     preprocessing_function=None,
    #     # image data format, either "channels_first" or "channels_last"
    #     data_format=None,
    #     # fraction of images reserved for validation (strictly between 0 and 1)
    #     validation_split=0.0
    # )

    # testDatagen = ImageDataGenerator(
    #     samplewise_std_normalization=True, rescale=1./255)

    # train_it = trainDatagen.flow_from_directory(
    #     './seti-data/primary_small/train', target_size=(224, 224), batch_size=BATCH_SIZE)
    # valid_it = testDatagen.flow_from_directory(
    #     './seti-data/primary_small/valid', target_size=(224, 224), batch_size=BATCH_SIZE)
    # test_it = testDatagen.flow_from_directory(
    #     './seti-data/primary_small/test', target_size=(224, 224), batch_size=BATCH_SIZE)

    # NUM_CLASSES = len(train_it.class_indices)

    trainPath = os.path.join("./seti-data/primary_small/train")
    valPath = os.path.join("./seti-data/primary_small/valid")
    testPath = os.path.join("./seti-data/primary_small/test")

    # parameter
    BATCH_SIZE = 128

    IMAGE_WIDTH = 96
    IMAGE_HEIGHT = 96

    (xTrain, yTrainStr) = loadData(trainPath,
                                   preprocess=[preprocess, IMAGE_WIDTH,  IMAGE_HEIGHT])
    (xVal, yValStr) = loadData(valPath, preprocess=[
        preprocess, IMAGE_WIDTH,  IMAGE_HEIGHT])
    (xTest, yTestStr) = loadData(testPath, preprocess=[
        preprocess, IMAGE_WIDTH,  IMAGE_HEIGHT])

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
    xTrain = xTrain.astype('float32') * 1./255
    xVal = xVal.astype('float32') * 1./255
    xTest = xTest.astype('float32') * 1./255

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
    EPOCHS = 200
    USE_AUGMENTATION = True
    DEPTH = COLORS * 6 + 2

    model = resnet_v1(input_shape=input_shape,
                      depth=DEPTH, num_classes=NUM_CLASSES)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])

    model.summary()

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    # start_time = time.time()
    checkpointer = ModelCheckpoint(
        filepath="./modelCheckpoin/weights.h5", verbose=1, save_best_only=True)
    callbacks = [lr_reducer, lr_scheduler, checkpointer]
    # model.fit_generator(train_it, epochs=EPOCHS,
    #                     validation_data=valid_it, callbacks=callbacks)

    # Run training, with or without data augmentation.
    if not USE_AUGMENTATION:
        print('Not using data augmentation.')
        model.fit(xTrain, yTrain,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(xVal, yVal),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
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
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
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
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(xTrain)

        with tensorflow.device('/gpu:0'):
            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(datagen.flow(xTrain, yTrain, batch_size=BATCH_SIZE),
                                validation_data=(xVal, yVal),
                                epochs=EPOCHS, verbose=1, workers=1,
                                callbacks=callbacks, use_multiprocessing=False)

    # elapsed_time = time.time() - start_time

    # scores = model.evaluate_generator(test_it, verbose=0)
    scores = model.evaluate(xTest, yTest, verbose=0)
    print(scores)
    model.save('./model/test_96_resV1.h5')
