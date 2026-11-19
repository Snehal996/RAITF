import json
import os

import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (128, 128)
NUM_LANDMARKS = 21

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="SignSense – Sign Language Recognition", page_icon="🤟", layout="centered")


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




@st.cache_resource(show_spinner="Loading model...")
def get_resources():
    return load_model(), load_labels(), make_detector()


model, labels, detector = get_resources()

st.title("🤟 SignSense")
st.caption("Sign language gesture recognition using MediaPipe hand landmarks and a CNN.")

with st.expander("How it works"):
    st.markdown(
        "1. **MediaPipe** detects up to two hands and 21 landmarks per hand (wrist, thumb, finger joints, fingertips).\n"
        "2. The landmarks are turned into numeric features and a 128×128 landmark image.\n"
        "3. A **supervised CNN**, trained on labelled gesture images, predicts the gesture.\n\n"
        f"Supported gestures ({len(labels)}): " + ", ".join(labels)
    )

mode = st.radio("Input", ["Take a photo", "Upload an image"], horizontal=True)
mirror = st.checkbox("Mirror image (try this if predictions look wrong)", value=False)

img_file = st.camera_input("Show your sign and take a photo") if mode == "Take a photo" \
    else st.file_uploader("Upload a photo of a sign", type=["jpg", "jpeg", "png"])

if img_file is not None:
    rgb = np.array(Image.open(img_file).convert("RGB"))
    if mirror:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
    out = predict(model, labels, detector, rgb)
    if out is None:
        st.warning("No hand detected. Use good lighting and keep your hand(s) fully in the frame.")
    else:
        name, conf = out["top"][0]
        st.success(f"Prediction: **{name}**  ({conf:.0%} confidence)")
        c1, c2 = st.columns(2)
        c1.image(out["annotated"], caption=f"Detected hands: {out['num_hands']}")
        c2.image(out["canvas"], caption="Landmark image given to the CNN")
        st.write("Top predictions")
        for n, p in out["top"]:
            st.progress(min(max(p, 0.0), 1.0), text=f"{n} – {p:.1%}")
