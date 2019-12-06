import numpy as np 

from resNetV1 import resnet_v1
from learningRateScheduler import lr_schedule
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau


if __name__ == "__main__":
    # trainPath = os.path.join("./seti-data/primary_small/train")
    # valPath = os.path.join("./seti-data/primary_small/valid")
    # testPath = os.path.join("./seti-data/primary_small/test")

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
        # rescale=None,
        rescale=1./255,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0
        )

    testDatagen = ImageDataGenerator(samplewise_std_normalization=True, rescale= 1./255)

    BATCH_SIZE = 32


    train_it = trainDatagen.flow_from_directory('./seti-data/primary_small/train', target_size=(224, 224), batch_size=BATCH_SIZE)
    valid_it = testDatagen.flow_from_directory('./seti-data/primary_small/valid', target_size=(224, 224), batch_size=BATCH_SIZE)
    test_it = testDatagen.flow_from_directory('./seti-data/primary_small/test', target_size=(224, 224), batch_size=BATCH_SIZE)

    NUM_CLASSES = len(train_it.class_indices)
    # print(NUM_CLASSES)
    
    # resnet V1
    EPOCHS = 200
    USE_AUGMENTATION = True
    DEPTH = 3 * 6 + 2
    model = resnet_v1(input_shape=(224,224,3), depth=DEPTH, num_classes=NUM_CLASSES)
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

    model.summary()

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)
    checkpointer = ModelCheckpoint(filepath="./modelCheckpoin/weights.h5", verbose=1, save_best_only=True)
    callbacks = [lr_reducer, lr_scheduler,checkpointer]
    model.fit_generator(train_it,epochs = EPOCHS, validation_data =valid_it, callbacks = callbacks)

    scores = model.evaluate_generator(test_it, verbose=0)
    print(scores)
    model.save('./model/my_model_resV1.h5')