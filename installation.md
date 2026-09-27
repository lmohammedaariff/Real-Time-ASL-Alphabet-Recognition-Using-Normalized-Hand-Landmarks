# Installation

Use Python 3.10–3.12. Create and activate a virtual environment, install `requirements.txt`, collect a balanced landmark CSV in `data/landmarks/landmarks.csv`, train, then run Streamlit or the OpenCV webcam application. MediaPipe platform support can vary; if its wheel is unavailable on a platform, webcam landmark extraction cannot run there, while the dataset/training utilities remain separable.

Webcam permissions must be granted to the local process/browser. Video is processed locally. Dataset collection stores only features after the explicit save button is clicked.
