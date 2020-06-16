from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout


def build_advanced_net(image_size=224, classes=7) -> Sequential:

    conv_base = ResNet50(include_top=False,
                         weights='imagenet',
                         input_shape=(image_size, image_size, 1))
    model = Sequential()
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    return model


def prep_finetune(model: Sequential) -> Sequential:

    for layer in model.layers[:-14]:
        layer.trainable = False

    return model
