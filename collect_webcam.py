"""Automatically collect a labeled batch from the local webcam.

For each run, choose the sign being shown. Only valid detected hands are saved;
frames are never written to disk. Repeat with a new label to build A-Z coverage.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from config import LETTERS, SETTINGS
from src.data.dataset_collector import append_sample, count_samples
from src.preprocessing.landmark_processor import extract_hand_features

LOGGER = logging.getLogger("signvision.collector")


def collect(label: str, target: int, interval: float, output: Path, camera: int) -> int:
    """Capture up to ``target`` valid samples for a single selected letter."""
    label = label.upper().strip()
    if label not in LETTERS:
        raise ValueError("Label must be one letter from A to Z")
    if target < 1 or interval <= 0:
        raise ValueError("Target must be positive and interval must be greater than zero")

    saved = count_samples(output, label)
    if saved >= target:
        return saved

    import cv2
    import mediapipe as mp

    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Webcam {camera} is unavailable. Check camera permissions or select another index.")

    last_sample = 0.0
    hands_api = mp.solutions.hands
    try:
        with hands_api.Hands(static_image_mode=False, max_num_hands=1,
                             min_detection_confidence=0.55,
                             min_tracking_confidence=0.5) as hands:
            while saved < target:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("The webcam stopped returning frames.")
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                now = time.monotonic()
                found_hand = bool(result.multi_hand_landmarks)
                if found_hand:
                    hand = result.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, hands_api.HAND_CONNECTIONS)
                    if now - last_sample >= interval:
                        try:
                            features = extract_hand_features(hand)
                            saved = append_sample(output, features, label)
                            last_sample = now
                        except ValueError as exc:
                            LOGGER.info("Skipping invalid hand landmarks: %s", exc)
                else:
                    cv2.putText(frame, "Show one complete hand", (18, 76),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

                cv2.putText(frame, f"Label {label} | collected {saved}/{target} | press q to stop",
                            (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 120), 2)
                cv2.imshow("SignVision batch collector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=LETTERS,
                        help="letter being signed during this capture batch")
    parser.add_argument("--target", type=int, default=SETTINGS.target_samples)
    parser.add_argument("--interval", type=float, default=0.25,
                        help="minimum seconds between saved examples")
    parser.add_argument("--camera", type=int, default=SETTINGS.camera_index)
    parser.add_argument("--output", type=Path, default=SETTINGS.dataset_path)
    args = parser.parse_args()
    try:
        saved = collect(args.label, args.target, args.interval, args.output, args.camera)
        print(f"Collection ended: {saved} samples saved for {args.label} to {args.output}")
    except (ImportError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Collection could not continue: {exc}\n")


if __name__ == "__main__":
    main()
