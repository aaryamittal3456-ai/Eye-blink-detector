"""
model_utils.py
--------------
Download and cache the MediaPipe FaceLandmarker model.
"""

import os
import urllib.request

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = "face_landmarker.task"


def ensure_model(status_callback=None) -> str:
    """
    Download the model if not already present.
    Calls status_callback(msg) with progress strings if provided.
    Returns the model path.
    """
    if not os.path.exists(MODEL_PATH):
        if status_callback:
            status_callback("Downloading FaceLandmarker model (~3 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        if status_callback:
            status_callback("Model downloaded ✓")
    return MODEL_PATH
