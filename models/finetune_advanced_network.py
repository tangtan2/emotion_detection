import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from models.util import load_images
from models.advanced_network import build_advanced_net, prep_finetune

if __name__ == '__main__':

    # parameters
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARN_RATE = 0.00001
    TEST_RATIO = 0.2
    SEED = 42

    # paths
    SAVED_MODEL = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/models/TBD'
    IMG_DIR = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/images'
    MODEL_DIR = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/models/'

    # load images from visgraf database

    # split images into training and validation sets

    # augment images using image data generator

    # create model and load previously trained weights then compile

    # fine tune model
