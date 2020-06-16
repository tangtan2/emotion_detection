import numpy as np
import os
import cv2
import json


# load images as arrays of pixel values
def load_images(folder: str, database: str, image_size=224) -> (np.array, np.array):

    # initialize empty arrays to hold images and labels
    images = []
    labels = []

    # iterate through images and get correct label from csv file
    image_props = json.load(open(folder + '/image_props.json', 'r'))
    for file in os.listdir(folder):
        if image_props['file']['database'] == database:
            file_path = folder + '/' + file
            image = cv2.imread(file_path)
            image = cv2.resize(image, image_size, image_size)
            label = image_props['file']['label']
            images.append(image)
            labels.append(label)

    # change to np arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# label and save image with predicted label
def label_save_image(image: np.array, label: str, savepath: str):

    # label image
    cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 225), 1)

    # save image to disk
    cv2.imwrite(savepath, image)


def draw_box(image: np.array, face: dict, colour=(0, 0, 0), text='', draw_face=False) -> np.array:

    image = cv2.rectangle(image,
                          (face['box'][0], face['box'][1]),
                          (face['box'][0] + face['box'][2], face['box'][1] + face['box'][3]),
                          color=colour,
                          thickness=2)
    if text != '':
        image = __write_text(image, text, (face['box'][0], face['box'][1]), colour)
    if draw_face:
        image = __draw_features(image, face['keypoints'], colour)
    return image


def __write_text(image: np.array, text: str, point: (int, int), colour=(0, 0, 0)) -> np.array:

    image = cv2.putText(image, text, point,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=10,
                        color=colour)
    return image


def __draw_features(image: np.array, features: dict, colour=(0, 0, 0)) -> np.array:

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
