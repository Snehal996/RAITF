# SignSense – Sign Language Recognition (Streamlit)

## Deploy free on Streamlit Community Cloud
1. Create a GitHub repo and push everything in this folder, with `app.py`, `requirements.txt` and `packages.txt` at the repo ROOT (keep the file names exactly; no "app (1).py") (`best_model.h5` is 11 MB, fine for GitHub).
   Do NOT push `cnn_dataset.pickle` (824 MB, over GitHub's 100 MB limit). `.gitignore` already excludes it.
2. Go to https://share.streamlit.io -> New app -> pick the repo, branch `main`, main file `app.py`.
3. Open **Advanced settings** and choose **Python 3.11** (or 3.12), then Deploy.
4. First build takes a few minutes (TensorFlow is large). The app sleeps when idle and wakes on visit.

## Run locally to test
    pip install -r requirements.txt
    streamlit run app.py
