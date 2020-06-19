import cv2
import numpy as np
import pandas as pd
import os
from models.advanced_network import build_advanced_net


def get_test_images(folder_path: str, image_size: int = 224):

    images = []
    labels = []
    file_ids = []

    print('[INFO] loading images...')
    label_df = pd.read_csv(folder_path + '/test_labels.csv')
    for file in os.listdir(folder_path + '/test'):
        if 'jpg' in file:
            file_path = folder_path + '/test/' + file
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (image_size, image_size)) / 255.0
            image = np.stack((image, image, image), axis=-1)
            image = np.expand_dims(image, axis=0)
            if file in label_df.image.values:
                images.append(image)
                labels.append(label_df.loc[label_df['image'] == file, 'emotion'].iloc[0])
                file_ids.append(file)
            if len(images) % 1000 == 0 and len(images) > 0:
                print(f'[INFO] {len(images)} images loaded...')
    print(f'[INFO] {len(images)} images loaded.')
    print(f'[INFO] each image has shape {images[0].shape}.')

    return images, labels, file_ids


if __name__ == '__main__':

    SAVED_ADV_MODEL_WEIGHTS = '/Users/tanyatang/Documents/Code/python/' \
                              'emotion_detection/data/advanced_models/june_18/model_e_04.hdf5'
    TEST_DIR = '/Users/tanyatang/Documents/Code/python/emotion_detection/data/images'

    labels_to_emotion = {0: 'anger',
                         1: 'disgust',
                         2: 'fear',
                         3: 'happy',
                         4: 'sadness',
                         5: 'surprise',
                         6: 'neutral'}

    adv_model = build_advanced_net(SAVED_ADV_MODEL_WEIGHTS)

    test_images, test_labels, test_file_ids = get_test_images(TEST_DIR)

    f = open(TEST_DIR + '/adv_test_preds.csv', mode='w')
    f.write('file_id,emotion,real_emotion,anger,disgust,fear,happiness,sadness,surprise,neutral\n')

    print('[INFO] making predictions for advanced model...')
    for i, test_image in enumerate(test_images):
        prediction = adv_model.predict(test_image)[0]
        emotion = labels_to_emotion[int(np.argmax(prediction))]
        real_emotion = test_labels[i]
        f.write(test_file_ids[i] + ',' + emotion +
                ',' + real_emotion + ',' +
                str(prediction[0]) + ',' +
                str(prediction[1]) + ',' +
                str(prediction[2]) + ',' +
                str(prediction[3]) + ',' +
                str(prediction[4]) + ',' +
                str(prediction[5]) + ',' +
                str(prediction[6]) + '\n')
    print('[INFO] predictions complete.')

    f.close()
