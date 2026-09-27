"""Per-browser WebRTC frame processor for continuous SignVision predictions."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import av
import cv2
import mediapipe as mp
from streamlit_webrtc import VideoProcessorBase

from config import SETTINGS
from src.inference.predictor import LandmarkPredictor
from src.inference.smoothing import PredictionSmoother
from src.preprocessing.landmark_processor import extract_hand_features

LOGGER = logging.getLogger("signvision.live_video")


class LivePredictionProcessor(VideoProcessorBase):
    """Process browser-camera frames locally and return an annotated live feed."""

    def __init__(self, model_path: str | Path | None = None):
        model_path = Path(model_path or (SETTINGS.model_dir / "best_model.joblib"))
        self.predictor = LandmarkPredictor(model_path)
        self.hands_api = mp.solutions.hands
        self.hands = self.hands_api.Hands(
            static_image_mode=False,
            max_num_hands=SETTINGS.num_hands,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.5,
        )
        self.smoother = PredictionSmoother(
            SETTINGS.prediction_window,
            SETTINGS.min_confidence,
            SETTINGS.stable_frames,
        )
        self.fps = 0.0
        self.last_error: str | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        started = time.perf_counter()
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        height, width = image.shape[:2]
        if width > 640:
            ratio = 640 / width
            image = cv2.resize(image, (640, int(height * ratio)), interpolation=cv2.INTER_AREA)
            height, width = image.shape[:2]

        result = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        label, confidence = "Show hand", 0.0
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand, self.hands_api.HAND_CONNECTIONS
            )
            xs = [int(point.x * width) for point in hand.landmark]
            ys = [int(point.y * height) for point in hand.landmark]
            margin = 14
            box = (max(0, min(xs) - margin), max(0, min(ys) - margin),
                   min(width - 1, max(xs) + margin), min(height - 1, max(ys) + margin))
            cv2.rectangle(image, box[:2], box[2:], (50, 220, 120), 2)
            try:
                raw_label, raw_confidence = self.predictor.predict(extract_hand_features(hand))
                stable = self.smoother.update(raw_label, raw_confidence)
                if stable:
                    label, confidence = stable
                else:
                    label, confidence = "Stabilizing…", raw_confidence
            except (ValueError, RuntimeError) as exc:
                self.last_error = str(exc)
                LOGGER.warning("Live frame prediction skipped: %s", exc)
                self.smoother.reset()
        else:
            self.smoother.update(None, 0.0)

        elapsed = max(time.perf_counter() - started, 1e-6)
        instant_fps = 1.0 / elapsed
        self.fps = instant_fps if self.fps == 0 else 0.85 * self.fps + 0.15 * instant_fps
        cv2.putText(image, f"Prediction: {label}", (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (40, 235, 60), 2, cv2.LINE_AA)
        cv2.putText(image, f"Confidence: {confidence:.1%}    FPS: {self.fps:.1f}",
                    (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (248, 248, 248), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def close(self) -> None:
        if getattr(self, "hands", None) is not None:
            self.hands.close()
            self.hands = None
