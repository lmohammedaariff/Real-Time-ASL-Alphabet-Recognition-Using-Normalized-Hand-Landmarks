# SignVision Project Report

**AI-Based Sign Language Alphabet Recognition**  
**Prepared for:** Mohammed Aariff L  
**Email:** lmohammedaariff@gmail.com  
**Report date:** 27 September 2026

## Project summary

SignVision is a Python computer-vision application for recognizing static American Sign Language (ASL) alphabet handshapes from a live camera stream. It detects one hand, converts the hand pose into 21 landmarks, normalizes the landmarks, and classifies the resulting 63 numeric features as a letter from A to Z. The project includes a Streamlit interface, webcam collection utilities, training and evaluation code, a real-time video processor, and a trained model for inference.

The model and processing pipeline are implemented and packaged. Live predictions require a browser or computer with an available camera and camera permission. The Codex session used to build this project did not expose a working webcam, so an end-to-end physical camera demonstration could not be confirmed here.

## Main capabilities

- Continuous webcam frame processing through the Streamlit real-time recognition page.
- Hand detection and drawing of landmarks and a hand bounding box.
- Normalized 21-point hand features and A–Z static handshape classification.
- Temporal prediction smoothing to reduce rapid changes in displayed letters.
- A word builder and optional speech support.
- Dataset import and landmark collection utilities.
- Training and comparison of Logistic Regression, SVM, Random Forest, and MLP classifiers.
- Evaluation reports, confusion matrices, and performance charts.
- Local-first processing: live video frames are not recorded by the application.

## Data and preparation

The training run used the public `Marxulia/asl_sign_languages_alphabets_v03` dataset. Its dataset page lists 10,873 labeled images. MediaPipe extracted usable hand landmarks from 8,333 images; 2,540 images were skipped because usable landmarks were not extracted. Each accepted sample was represented by 21 three-dimensional landmarks (63 values), normalized relative to the wrist and hand extent.

The dataset card did not state an explicit image license. To avoid redistributing the original images or derived sample data without confirmed terms, the ZIP contains the data import/collection code and the trained model, but not the source images or generated landmark CSV. The data can be regenerated locally using the documented import command after checking the dataset terms.

## Model evaluation

A stratified 80/20 holdout evaluation selected the SVM as the best model by macro F1. The run used 8,333 samples across the 26 letters.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 81.34% | 81.38% | 81.28% | 81.17% |
| SVM (selected) | 89.62% | 89.57% | 89.46% | 89.41% |
| Random Forest | 87.40% | 87.52% | 87.27% | 87.22% |
| MLP | 89.56% | 89.63% | 89.41% | 89.41% |

The selected SVM trained in approximately 5.76 seconds. Its measured classifier inference latency was approximately 0.446 milliseconds per sample on the development environment. This measurement covers model inference, not the complete webcam pipeline. The holdout split is sample-stratified rather than signer-independent, so these scores do not establish accuracy for new signers, environments, or camera setups.

## Running the project

Use Python 3.10–3.12. From the extracted `SignVision` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the **Real-Time Recognition** page, allow camera access, and show one alphabet handshape to the camera. The page displays the predicted letter with confidence and overlays the detected hand landmarks. The local OpenCV alternative is:

```powershell
python -m src.inference.realtime
```

To regenerate the training landmark data, review the dataset terms and run:

```powershell
python -m src.data.import_huggingface_dataset
python -m src.training.train
```

For camera-based sample collection, see the commands and instructions in `README.md` and `docs/`.

## Verification and known scope

The project test suite passed **12 tests** during implementation. Python source compilation also completed successfully. These checks cover project behavior but do not substitute for testing with an available physical webcam.

This application classifies static alphabet handshapes. It is not a full sign-language translator. The letters J and Z involve motion and are not reliably represented by a single static pose. Signer-independent testing, more varied collection conditions, and motion-aware models are recommended future work.

## Files in the ZIP

The archive contains the application, source modules, documentation, tests, dependency list, evaluation outputs, and the selected trained SVM model. It excludes virtual environments, temporary files, raw source images, generated landmark data, and redundant large model artifacts.
