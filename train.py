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
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        vertical_flip=True,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    testDatagen = ImageDataGenerator(
        rescale=1./255)

    BATCH_SIZE = 16

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