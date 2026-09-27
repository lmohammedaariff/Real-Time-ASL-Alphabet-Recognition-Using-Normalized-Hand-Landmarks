"""CSV-backed collector used by webcam tools and scripts."""
from pathlib import Path
import csv
import logging
from config import LETTERS
from src.preprocessing.landmark_processor import FEATURE_COUNT

LOGGER = logging.getLogger(__name__)

def append_feature_row(path: str | Path, features, label: str) -> None:
    """Append one validated vector efficiently without recounting the whole CSV."""
    label = str(label).upper().strip()
    if label not in LETTERS:
        raise ValueError("Label must be an uppercase letter A-Z")
    values = list(features)
    if len(values) != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features")
    import numpy as np
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Feature values must be finite")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [f"f{i}" for i in range(FEATURE_COUNT)] + ["label"]
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow([*arr.tolist(), label])
    LOGGER.info("Appended sample for %s to %s", label, path)

def append_sample(path: str | Path, features, label: str) -> int:
    """Append one sample and return its class count (convenient for UI use)."""
    append_feature_row(path, features, label)
    return count_samples(path, label)

def count_samples(path: str | Path, label: str | None = None) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    import pandas as pd
    data = pd.read_csv(path)
    if "label" not in data:
        return 0
    return int((data.label == label).sum()) if label else len(data)
