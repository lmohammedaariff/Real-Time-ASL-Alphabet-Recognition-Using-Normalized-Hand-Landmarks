"""OpenCV + MediaPipe real-time A-Z recognition (press q to quit)."""
import time
import cv2
import mediapipe as mp
from config import SETTINGS
from src.inference.predictor import LandmarkPredictor
from src.inference.smoothing import PredictionSmoother
from src.preprocessing.landmark_processor import extract_hand_features

def main():
    model_path = SETTINGS.model_dir / "best_model.joblib"
    try:
        predictor = LandmarkPredictor(model_path)
    except Exception as exc:
        raise SystemExit(f"Could not load trained model ({exc}). Run python -m src.training.train first.")
    cap = cv2.VideoCapture(SETTINGS.camera_index)
    if not cap.isOpened():
        raise SystemExit("Webcam unavailable. Check camera permissions or camera_index in config.py.")
    hands_api = mp.solutions.hands
    smoother = PredictionSmoother(SETTINGS.prediction_window, SETTINGS.min_confidence, SETTINGS.stable_frames)
    fps = 0.0
    try:
        with hands_api.Hands(max_num_hands=SETTINGS.num_hands, min_detection_confidence=0.55,
                             min_tracking_confidence=0.5) as hands:
            while True:
                start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame could not be read.")
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                display, confidence = "—", 0.0
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, hands_api.HAND_CONNECTIONS)
                    height, width = frame.shape[:2]
                    xs = [int(point.x * width) for point in hand.landmark]
                    ys = [int(point.y * height) for point in hand.landmark]
                    margin = 18
                    x1, x2 = max(0, min(xs) - margin), min(width - 1, max(xs) + margin)
                    y1, y2 = max(0, min(ys) - margin), min(height - 1, max(ys) + margin)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 120), 2)
                    try:
                        display, confidence = predictor.predict(extract_hand_features(hand))
                    except ValueError:
                        pass
                stable = smoother.update(display if display != "—" else None, confidence)
                if stable:
                    display, confidence = stable
                else:
                    display = "…"
                fps = 0.9 * fps + 0.1 / max(time.perf_counter() - start, 1e-6)
                cv2.putText(frame, f"Prediction: {display}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (40, 235, 60), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Confidence: {confidence:.1%}    FPS: {fps:.1f}",
                            (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
                cv2.imshow("SignVision — press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
