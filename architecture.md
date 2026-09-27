# Architecture

SignVision separates capture, landmark preprocessing, training, inference, and interface layers. A capture adapter obtains frames locally; MediaPipe Hands returns 21 landmarks. `normalize_landmarks` validates finite values, subtracts the wrist coordinate, and divides by the largest centered coordinate magnitude. It returns a reusable 63-value vector.

The classifier estimates one static alphabet class. `PredictionSmoother` rejects weak frames and requires a majority of recent accepted observations before showing a stable label. `WordBuilder` only appends on an explicit accept action; the prototype keeps these operations separate from continuous frame inference to prevent duplicate characters.

The Streamlit live-recognition page streams browser frames over local WebRTC to a per-session processor. It runs MediaPipe and the trained classifier, then sends the annotated frame back for display. The collector writes features only after an explicit save action. Both browser live recognition and the OpenCV inference command keep frames in process memory and never record them.
