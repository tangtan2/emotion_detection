import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from models.util import load_images
from models.advanced_network import build_advanced_net

if __name__ == '__main__':

    # parameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARN_RATE = 0.001
    TEST_RATIO = 0.2
    SEED = 42

    # paths
    IMG_DIR = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/images'
    MODEL_DIR = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/models'

    # load images from fer and ck databases
    fer_images, fer_labels = load_images(IMG_DIR, 'fer')
    ck_images, ck_labels = load_images(IMG_DIR, 'ck')
    images = np.append(fer_images, ck_images)
    labels = np.append(fer_labels, ck_labels)

    # split images into training and validation sets
    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=TEST_RATIO, random_state=SEED)

    # augment images using image data generator
    aug = ImageDataGenerator(rotation_range=0.2,
                             shear_range=0.15)

    # create and compile model
    model = build_advanced_net()
    model.summary()
    optimizer = Adam(learning_rate=LEARN_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # train model
    ES = EarlyStopping(monitor='val_loss',
                       patience=2)
    MC = ModelCheckpoint(MODEL_DIR + '/stage_1_model_e_{epoch:02d}.hdf5',
                         save_best_only=True,
                         save_weights_only=True)
    model.fit(aug.flow(train_x, train_y, batch_size=BATCH_SIZE),
              epochs=NUM_EPOCHS,
              validation_data=aug.flow(test_x, test_y, batch_size=BATCH_SIZE),
              callbacks=[ES, MC],
              verbose=1)
