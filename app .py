import numpy as np
import streamlit as st
from PIL import Image

import sign_engine as eng

st.set_page_config(page_title="SignSense – Sign Language Recognition", page_icon="🤟", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_resources():
    return eng.load_model(), eng.load_labels(), eng.make_detector()


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
    out = eng.predict(model, labels, detector, rgb)
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
