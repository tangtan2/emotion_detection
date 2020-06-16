from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization


def build_basic_net(width: int, height: int, depth: int, classes: int) -> Sequential:

    # initialize model with input shape
    model = Sequential()

    # convolution base
    model.add(Conv2D(32, (5, 5),
                     padding='same',
                     input_shape=(height, width, depth),
                     activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5),
                     activation='relu',
                     padding='same'))
    model.add(Conv2D(128, (5, 5),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same'))
    model.add(Conv2D(256, (3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(classes, activation='softmax'))

    return model
