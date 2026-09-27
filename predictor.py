"""Model loading and classification."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

class LandmarkPredictor:
    def __init__(self, model_path: str | Path):
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict) and "model" in bundle:
            self.model, self.labels = bundle["model"], list(bundle["labels"])
        else:
            self.model, self.labels = bundle, list(getattr(bundle, "classes_", []))
        if not self.labels:
            raise ValueError("Model bundle does not expose class labels")

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        values = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if values.shape[1] != 63 or not np.isfinite(values).all():
            raise ValueError("Expected 63 finite landmark features")
        x = pd.DataFrame(values, columns=[f"f{i}" for i in range(63)])
        probs = self.model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        model_label = self.model.classes_[idx]
        if isinstance(model_label, (int, np.integer)) and 0 <= int(model_label) < len(self.labels):
            label = self.labels[int(model_label)]
        else:
            label = str(model_label)
        return label, float(probs[idx])
