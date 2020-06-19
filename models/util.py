import numpy as np
import os
import cv2
import json


# load images as arrays of pixel values
def load_images(folder_path: str, databases: list, image_size: int = 224) -> (np.array, np.array):

    # initialize empty arrays to hold images and labels
    images = []
    labels = []

    # iterate through images and get correct label from csv file
    print('[INFO] loading images...')
    image_props = json.load(open(folder_path + '/image_props.json', 'r'))
    images_folder = folder_path + '/images'
    for file in os.listdir(images_folder):
        if 'png' in file and image_props[file]['database'] in databases:
            file_path = images_folder + '/' + file
            image = cv2.imread(file_path)
            image = cv2.resize(image, (image_size, image_size)) / 255.0
            label = image_props[file]['label']
            images.append(image)
            labels.append(label)
            if len(images) % 1000 == 0:
                print(f'[INFO] {len(images)} images loaded...')
    print(f'[INFO] {len(images)} images loaded.')

    # convert to numpy arrays
    images = np.array(images)
    print(f'[INFO] images array has shape {images.shape}')
    labels = np.array(labels)

    return images, labels


def draw_box(image: np.array, face: dict, colour: (int, int, int) = (0, 0, 0), emotion: str = '',
             probability: str = '', draw_face: bool = False) -> np.array:

    image = cv2.rectangle(image,
                          (face['box'][0], face['box'][1]),
                          (face['box'][0] + face['box'][2], face['box'][1] + face['box'][3]),
                          color=colour,
                          thickness=2)
    if emotion != '':
        image = __write_text(image,
                             'Emotion: ' + emotion + ' Likelihood: ' + probability + '%',
                             (face['box'][0], face['box'][1] - 10),
                             colour=colour)
    if draw_face:
        image = __draw_features(image,
                                face['keypoints'],
                                colour=colour)

    return image


def __write_text(image: np.array, text: str, point: (int, int), colour: (int, int, int) = (255, 255, 255)) -> np.array:

    image = cv2.putText(image, text, point,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        thickness=2,
                        color=colour)

    return image


def __draw_features(image: np.array, features: dict, colour: (int, int, int) = (0, 0, 0)) -> np.array:

    image = cv2.circle(image,
                       features['left_eye'],
                       radius=3,
                       color=colour,
                       thickness=-1)
    image = cv2.circle(image,
                       features['right_eye'],
                       radius=3,
                       color=colour,
                       thickness=-1)
    image = cv2.circle(image,
                       features['nose'],
                       radius=3,
                       color=colour,
                       thickness=-1)
    image = cv2.circle(image,
                       features['mouth_left'],
                       radius=3,
                       color=colour,
                       thickness=-1)
    image = cv2.circle(image,
                       features['mouth_right'],
                       radius=3,
                       color=colour,
                       thickness=-1)

    return image
