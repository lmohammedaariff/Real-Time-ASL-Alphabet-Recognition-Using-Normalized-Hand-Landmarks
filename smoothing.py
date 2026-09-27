"""Confidence-aware temporal prediction stabilization."""
from collections import deque, Counter

class PredictionSmoother:
    def __init__(self, window_size=10, min_confidence=0.70, stable_frames=5):
        if window_size < 1 or stable_frames < 1 or stable_frames > window_size:
            raise ValueError("Require window_size >= stable_frames >= 1")
        self.window = deque(maxlen=window_size)
        self.min_confidence = min_confidence
        self.stable_frames = stable_frames

    def update(self, label: str | None, confidence: float):
        if label is None or confidence < self.min_confidence:
            self.window.clear()
            return None
        self.window.append((label, confidence))
        counts = Counter(label for label, _ in self.window)
        best, n = counts.most_common(1)[0]
        if n < self.stable_frames:
            return None
        mean_conf = sum(c for l, c in self.window if l == best) / n
        return best, mean_conf

    def reset(self):
        self.window.clear()
