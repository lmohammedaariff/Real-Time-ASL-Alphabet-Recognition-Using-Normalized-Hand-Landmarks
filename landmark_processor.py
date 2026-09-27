"""Validation and translation/scale normalization for MediaPipe hand landmarks."""
from __future__ import annotations
import numpy as np

LANDMARK_COUNT = 21
FEATURE_COUNT = LANDMARK_COUNT * 3

def normalize_landmarks(points: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    """Return a translation- and scale-invariant flattened 63-value feature.

    Accepts (21, 2) or (21, 3) finite coordinates. Scale is the largest
    absolute centered coordinate, which remains defined for a closed hand.
    """
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape == (LANDMARK_COUNT, 2):
        arr = np.column_stack((arr, np.zeros(LANDMARK_COUNT, dtype=np.float32)))
    if arr.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected (21, 2) or (21, 3) landmarks, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Landmarks contain missing or non-finite coordinates")
    centered = arr - arr[0]
    scale = float(np.max(np.abs(centered)))
    if scale <= epsilon:
        raise ValueError("Landmarks have zero spatial extent")
    normalized = centered / scale
    return normalized.reshape(FEATURE_COUNT).astype(np.float32)

def extract_hand_features(hand_landmarks) -> np.ndarray:
    """Convert a MediaPipe NormalizedLandmarkList to normalized features."""
    if hand_landmarks is None or len(hand_landmarks.landmark) != LANDMARK_COUNT:
        raise ValueError("A complete set of 21 hand landmarks is required")
    points = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    return normalize_landmarks(points)
