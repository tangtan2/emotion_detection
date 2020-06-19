import time
import numpy as np
from mtcnn import MTCNN
import cv2
from tensorflow.keras.models import Model
from models.advanced_network import build_advanced_net
from models.util import draw_box


class FaceDetector:

    def __init__(self):
        self.detector = MTCNN()

    def get_faces(self, image: np.array) -> (list, dict):
        faces = self.detector.detect_faces(image)
        face_images = [image[face['box'][1]:face['box'][1] + face['box'][3],
                       face['box'][0]:face['box'][0] + face['box'][2]] for face in faces]
        return face_images, faces


class EmotionPredictor:

    def __init__(self, model: Model, image_size: int = 224):
        self.predictor = model
        self.emotions = {0: 'anger',
                         1: 'disgust',
                         2: 'fear',
                         3: 'happy',
                         4: 'sadness',
                         5: 'surprise',
                         6: 'neutral'}
        self.image_size = image_size

    def predict_image(self, image: np.array) -> (str, float):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.image_size, self.image_size)) / 255.0
        image = np.stack((image, image, image), axis=-1)
        image = np.expand_dims(image, axis=0)
        prediction = self.predictor.predict(image)
        return self.emotions[int(np.argmax(prediction[0]))], max(prediction[0]), prediction


class DynamicPredictor:

    def __init__(self, model: Model):
        self.face_detector = FaceDetector()
        self.emotion_predictor = EmotionPredictor(model)
        self.cap = cv2.VideoCapture(0)
        self.colour = {'anger': (255, 0, 0),
                       'disgust': (0, 204, 102),
                       'fear': (255, 128, 0),
                       'happy': (255, 255, 0),
                       'sadness': (0, 128, 255),
                       'surprise': (204, 0, 204),
                       'neutral': (255, 255, 255)}

    def start_capture(self):
        print('[INFO] starting camera...')
        timer = time.time()
        last_emotion = ''
        last_prob = 0.0
        while True:
            _, frame = self.cap.read()
            cv2.putText(frame,
                        'Press Q to quit',
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        thickness=2,
                        color=(0, 0, 0),
                        org=(50, 50))
            images, faces = self.face_detector.get_faces(frame)
            if time.time() > timer + 1:
                if len(images) > 0:
                    last_emotion, last_prob, all_probs = self.make_prediction(images[0])
                    print(f'[INFO] updated emotion to {last_emotion}...')
                    frame = draw_box(frame,
                                     faces[0],
                                     colour=self.colour[last_emotion],
                                     emotion=last_emotion,
                                     probability=str(round(last_prob * 100, 2)))
                timer = time.time()
            else:
                if last_emotion != '' and len(images) > 0:
                    frame = draw_box(frame,
                                     faces[0],
                                     colour=self.colour[last_emotion],
                                     emotion=last_emotion,
                                     probability=str(round(last_prob * 100, 2)))
                elif len(images) > 0:
                    frame = draw_box(frame,
                                     faces[0],
                                     colour=(0, 0, 0),
                                     emotion='None',
                                     probability='0')
            cv2.imshow('Emotion Predictor', frame)
            if cv2.waitKey(1) == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                break
        print('[INFO] program ended.')

    def make_prediction(self, face: np.array) -> np.array:
        emotion, probability, all_probs = self.emotion_predictor.predict_image(face)
        return emotion, probability, all_probs


if __name__ == '__main__':

    SAVED_MODEL_WEIGHTS = '/Users/tanyatang/Documents/Code/python/emotion_detection/' \
                          'data/advanced_models/june_18/model_e_04.hdf5'

    network = build_advanced_net()
    run = DynamicPredictor(network)
    run.start_capture()
