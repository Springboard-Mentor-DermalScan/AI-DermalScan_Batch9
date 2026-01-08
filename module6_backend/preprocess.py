import cv2
import numpy as np
from .config import IMAGE_SIZE

def preprocess_face(face_img):
    face = cv2.resize(face_img, IMAGE_SIZE)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face
