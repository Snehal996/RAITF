"""Core pipeline: image -> MediaPipe landmarks -> (features, landmark image) -> CNN prediction.
Mirrors createDataCNN.py exactly so inference matches training."""
import json
import os

import cv2
import mediapipe as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (128, 128)
NUM_LANDMARKS = 21

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def load_labels():
    with open(os.path.join(HERE, "labels.json")) as f:
        return json.load(f)


def load_model():
    import keras
    return keras.models.load_model(os.path.join(HERE, "best_model.h5"), compile=False)


def make_detector():
    # Same settings as dataset creation
    return mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)


def extract_features(hand_list):
    """(2, 21, 3) array; hands sorted left-to-right by mean x, unused slot = zeros."""
    features = np.zeros((2, NUM_LANDMARKS, 3), dtype=np.float32)
    hands_data = sorted(
        ((np.mean([lm.x for lm in h.landmark]), h) for h in hand_list), key=lambda t: t[0]
    )
    for idx, (_, h) in enumerate(hands_data[:2]):
        for i, lm in enumerate(h.landmark):
            features[idx, i] = [lm.x, lm.y, lm.z]
    return features


def landmark_canvas(hand_list):
    """128x128 black canvas with landmarks drawn (RGB), as in training."""
    canvas = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    for h in hand_list:
        mp_drawing.draw_landmarks(
            canvas, h, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
            mp_drawing.DrawingSpec(color=(100, 255, 100), thickness=2),
        )
    return canvas


def predict(model, labels, detector, rgb_image, top_k=3):
    """rgb_image: HxWx3 uint8 RGB. Returns dict or None if no hand found."""
    results = detector.process(rgb_image)
    if not results.multi_hand_landmarks:
        return None
    hands = results.multi_hand_landmarks
    feats = extract_features(hands)
    canvas = landmark_canvas(hands)
    # Model was trained on raw 0-255 landmark images (no /255 scaling) - verified on the dataset.
    probs = model.predict([feats[None], canvas[None].astype("float32")], verbose=0)[0]
    order = np.argsort(probs)[::-1][:top_k]
    # Annotated view of the full photo for display
    annotated = rgb_image.copy()
    for h in hands:
        mp_drawing.draw_landmarks(annotated, h, mp_hands.HAND_CONNECTIONS)
    return {
        "top": [(labels[i], float(probs[i])) for i in order],
        "canvas": canvas,
        "annotated": annotated,
        "num_hands": len(hands),
    }
