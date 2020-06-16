import numpy as np
from mtcnn import MTCNN
import cv2
from tensorflow.keras.models import Model


class FaceDetector:

    def __init__(self, model: MTCNN):
        self.detector = model

    def are_faces_present(self, image: np.array) -> bool:
        faces = self.detector.detect_faces(image)
        return (faces is not None) and (len(faces) > 0)

    def get_faces(self, image: np.array) -> (list, list):
        faces = self.detector.detect_faces(image)
        face_images = [image[face['box'][0]:face['box'][0] + face['box'][2],
                       face['box'][1]:face['box'][1] + face['box'][3]] for face in faces]
        return face_images, [face['box'] for face in faces]


class EmotionPredictor:

    def __init__(self, model: Model, labels: list, image_size=224):
        self.predictor = model
        self.emotions = labels
        self.image_size = image_size

    def predict_image(self, image: np.array) -> (str, float):
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image) / 255.0
        prediction = self.predictor.predict(image)
        return self.emotions[np.argmax(prediction)[0]], max(prediction)


class DynamicPredictor:

    def __init__(self):
        a = 0  # TBD

    def start_capture(self):
        a = 0  # TBD

    def refresh(self):
        a = 0  # TBD

    def make_prediction(self):
        a = 0  # TBD


if __name__ == '__main__':
    b = 0  # TBD
