import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "module5_streamlit",
        "model",
        "mobilenetv2_module3.h5"
    )
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "clear_skin",
    "dark_spots",
    "wrinkles",
    "puffy_eyes"
]

LOG_FILE = os.path.join(BASE_DIR, "predictions.log")
