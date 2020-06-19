from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization


def build_basic_net(saved_model=None, image_size=224, classes=7) -> Sequential:

    if saved_model is not None:
        model = load_model(saved_model)
    else:
        # initialize model with input shape
        model = Sequential()

        # convolution base
        model.add(Conv2D(8, (5, 5),
                         padding='same',
                         input_shape=(image_size, image_size, 1),
                         activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (5, 5),
                         activation='relu',
                         padding='same'))
        model.add(Conv2D(16, (5, 5),
                         activation='relu',
                         padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(64, (3, 3),
                         activation='relu',
                         padding='same'))
        model.add(Conv2D(64, (3, 3),
                         activation='relu',
                         padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # fully connected layers
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(classes, activation='softmax'))

    return model
