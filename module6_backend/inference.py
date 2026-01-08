import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D

from .config import MODEL_PATH, CLASS_NAMES
from .preprocess import preprocess_face


class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


_model = None


def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={
                "DepthwiseConv2D": PatchedDepthwiseConv2D
            }
        )
    return _model


def predict(face_img):
    model = load_model()
    processed = preprocess_face(face_img)

    preds = model.predict(processed, verbose=0)[0]

    pred_index = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    return {
        "label": CLASS_NAMES[pred_index],
        "confidence": confidence,
        "raw_scores": preds.tolist()
    }
