from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models.util import load_images
from models.basic_network import build_basic_net

if __name__ == '__main__':

    # parameters
    NUM_EPOCHS = 10
    BATCH_SIZE = 128
    LEARN_RATE = 0.001
    TEST_RATIO = 0.2
    NUM_CLASSES = 7
    SEED = 42

    # paths
    IMG_DIR = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/images'
    MODEL_DIR = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/basic_models'

    # load images from all databases
    images, labels = load_images(IMG_DIR, ['fer', 'ck', 'visgraf'], image_size=112)
    labels = to_categorical(labels, num_classes=NUM_CLASSES)

    # split images into training and validation sets
    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=TEST_RATIO, random_state=SEED)

    # augment images using image data generator
    aug = ImageDataGenerator(rotation_range=0.2,
                             shear_range=0.15)

    # create and compile model
    print('[INFO] creating model...')
    model = build_basic_net()
    model.summary()
    optimizer = Adam(learning_rate=LEARN_RATE,
                     decay=LEARN_RATE / (NUM_EPOCHS * 0.5))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print('[INFO] model compiled.')

    # train model
    ES = EarlyStopping(monitor='val_loss',
                       patience=2)
    print('[INFO] starting training...')
    H = model.fit(aug.flow(train_x, train_y, batch_size=BATCH_SIZE),
                  epochs=NUM_EPOCHS,
                  validation_data=aug.flow(test_x, test_y, batch_size=BATCH_SIZE),
                  callbacks=[ES],
                  verbose=1)
    model.save(MODEL_DIR + '/trained_model.hdf5')
    print('[INFO] model trained and saved.')

    # plot loss and accuracy history
    plt.figure()
    plt.plot(H.history['accuracy'])
    plt.plot(H.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(MODEL_DIR + '/model_accuracy.png')
    plt.figure()
    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(MODEL_DIR + '/model_loss.png')
