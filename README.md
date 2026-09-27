# SignVision

**Real-Time Sign Language Alphabet Recognition System** — an end-to-end, landmark-first ASL fingerspelling portfolio project.

## Overview

SignVision captures a hand pose, extracts MediaPipe's 21 landmarks, normalizes them into a 63-value feature vector, and classifies a static alphabet handshape. The app includes local webcam inference, temporal smoothing, a word buffer, custom landmark collection, four-model comparison, reports, and a Streamlit dashboard.

## Problem Statement and Objectives

Alphabet fingerspelling is a useful constrained computer-vision task, but raw-frame predictions flicker and vary with hand size and position. SignVision provides reusable preprocessing and a reproducible, lightweight classifier pipeline, while making the scope and data limitations explicit.

## Features

- Reusable landmark validation, wrist-relative translation, scale normalization, and flattening.
- Browser-based custom feature collection to CSV and OpenCV real-time inference.
- Logistic Regression, SVM, Random Forest, and MLP comparison.
- Accuracy, macro precision/recall/F1, per-class report, confusion matrices, training time and latency.
- Confidence threshold and configurable temporal vote; word-builder helper with cooldown.
- Streamlit home, recognition instructions, collection, evaluation, and about pages.
- Local-first webcam processing; no frame storage unless a user explicitly opts into a separate capture workflow (the included collector stores landmarks only).

## System Architecture

```text
Local camera → MediaPipe Hands → 21 × (x,y,z) → wrist/scale normalization
             → sklearn classifier → confidence filter → temporal vote → letter
             → explicit word controls / optional offline speech extension
```

## Technology Stack

Python, OpenCV, MediaPipe, NumPy, Pandas, scikit-learn, Joblib, Matplotlib, Seaborn, Streamlit, pytest. Dependencies are listed in `requirements.txt`.

## Dataset

Training consumes `data/landmarks/landmarks.csv` with 63 numeric feature fields `f0`–`f62` and `label` A–Z. The starter public split can be extracted with `python -m src.data.import_huggingface_dataset`; the generated landmark CSV and model are local data and excluded from Git by default. Collect additional examples across different signers and sessions for better real-world generalization. Static Sign Language MNIST images are not equivalent to landmark features and require a conversion/detection step; they are not silently loaded as features.

## Data Preprocessing

The shared preprocessing module requires 21 finite landmarks, subtracts the wrist coordinate, scales by maximum absolute centered extent, then flattens in landmark order. It is used by both collector and webcam inference. Rotation normalization is not enabled because orientation may carry class information.

## Model Architecture and Model Comparison

The MLP has two hidden layers (128, 64), ReLU activations, and sklearn early stopping. Logistic Regression, SVM, and MLP are scaled in leakage-safe pipelines; Random Forest operates on normalized coordinates. Train/test splitting is stratified. Best macro-F1 is selected. Model-comparison metrics are generated from actual training data; no results are fabricated in the repository.

## Results

On the extracted public split, the recorded stratified 80/20 holdout run selected SVM: accuracy **89.62%**, macro precision **89.57%**, macro recall **89.46%**, and macro F1 **89.41%**. The MLP scored 89.56% accuracy and 89.41% macro F1. The best SVM trained in 5.76 seconds with a measured batch-average inference latency of 0.446 ms per sample in this environment. The run produced `reports/model_comparison.csv`, class report, confusion matrices, and charts. This is a sample-stratified split from one public dataset, not signer-independent evidence; test with your own camera and signers before making generalization claims. The dataset card has no explicit image license, so its source and terms must be checked before redistribution.

## Real-Time Demo

Open the Streamlit **Real-Time Recognition** page and click **START** to stream webcam frames continuously in the browser. The annotated feed shows landmarks, a hand box, stabilized prediction, confidence, and FPS. Frames are processed locally and never recorded. Alternatively, run `python -m src.inference.realtime` for the OpenCV desktop window (press `q` to quit). J and Z require motion and remain limitations of this static-pose model.

## Installation and Usage

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### Dataset Collection

Open the Dataset Collection page, choose a letter, capture a frame, then click **Save this landmark sample**. A complete hand must be visible. Repeat across letters, people, and sessions. Only normalized features and the label are stored.

For automatic interval collection from a local webcam, run one batch per letter:

```powershell
python -m src.data.collect_webcam --label A --target 500 --interval 0.25
```

The preview shows detection landmarks and progress. It skips frames without a valid hand; press `q` to stop. To import a downloaded image dataset organized into A–Z folders, extract landmark features with:

```powershell
python -m src.data.import_image_dataset --input "path\to\asl_alphabet_train"
```

The importer can handle nested splits, reports imported and skipped counts, and stores features rather than source images. Ensure the dataset is appropriately licensed and that its hands can be detected by MediaPipe.

To download and extract the public A–Z starter split directly, run:

```powershell
python -m src.data.import_huggingface_dataset
```

This processes the public [Marxulia ASL alphabet dataset](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03) (10,873 labeled images shown on its dataset page), keeps the parquet cache locally, and writes only detected hand landmarks and labels to the project CSV. The dataset card has no explicit image license listed; check its terms/provenance before redistribution. See `reports/dataset_extraction.json` for the actual extracted/skipped counts.

### Model Training

```powershell
python -m src.training.train
# optional paths
python -m src.training.train --dataset data/landmarks/landmarks.csv --models models --reports reports
```

### Running the Streamlit App

```powershell
streamlit run app.py
```

### Running webcam inference

```powershell
python -m src.inference.realtime
```

## Project Structure

```text
SignVision/
├── app.py, config.py, requirements.txt
├── data/{raw,processed,landmarks}/
├── docs/{architecture,model,dataset,installation}.md
├── models/                 # generated locally
├── reports/                # generated locally
├── src/{data,preprocessing,training,inference,utils,word_builder}/
└── tests/
```

## Screenshots

Capture screenshots locally after launching Streamlit; no screenshots are checked in.

## Limitations

This is static alphabet fingerspelling recognition, not full sign-language translation. J/Z movement, two-hand signs, facial grammar, transitions, signer-independent evaluation, and robust occlusion handling need temporal and broader data. The collector requires MediaPipe support on the host. Browser camera availability depends on permission and deployment context.

## Future Improvements

Add signer-grouped splits, landmark quality checks, temporal sequence models for J/Z, calibration, model cards, reproducible dataset manifests, benchmarking on a documented public dataset, and a native live video component in Streamlit.

## Research Scope

Potential experiments include landmark versus image models, signer-independent splits, background/lighting robustness, cross-user generalization, latency optimization, temporal signs, and continuous recognition. These are research directions, not claimed results or novelty.

## Privacy

Live inference processes frames locally and does not record or upload them. The collector stores only landmark feature rows after the user explicitly clicks save. Confirm consent before collecting samples from other people.

## Tests

```powershell
python -m pytest
```

## How to Run SignVision

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# collect samples in the Streamlit Dataset Collection page
python -m src.training.train
python -m src.inference.realtime
streamlit run app.py
```

## Author

Add your name, affiliation, and contact details before publishing this portfolio project.
