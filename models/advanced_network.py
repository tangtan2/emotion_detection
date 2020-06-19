from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization


def build_advanced_net(model_weights=None, image_size: int = 224, classes: int = 7) -> Sequential:

    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=(image_size, image_size, 3))
    for layer in conv_base.layers[:-2]:
        layer.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    if model_weights is not None:
        model.load_weights(model_weights)

    return model
