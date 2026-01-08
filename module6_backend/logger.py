from datetime import datetime
from .config import LOG_FILE

def log_prediction(label, confidence, bbox):
    with open(LOG_FILE, "a") as f:
        f.write(
            f"{datetime.now()} | "
            f"Label: {label} | "
            f"Confidence: {confidence:.2f}% | "
            f"BBox: {bbox}\n"
        )
