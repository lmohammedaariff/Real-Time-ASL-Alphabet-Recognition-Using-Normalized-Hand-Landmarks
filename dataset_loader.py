"""Load, validate, and split landmark datasets."""
from pathlib import Path
import logging
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from config import LETTERS
from src.preprocessing.landmark_processor import FEATURE_COUNT

LOGGER = logging.getLogger(__name__)

def load_landmark_csv(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Landmark dataset not found: {path}. Collect samples first.")
    frame = pd.read_csv(path)
    if "label" not in frame:
        raise ValueError("CSV must contain a 'label' column")
    feature_cols = [c for c in frame.columns if c != "label"]
    if len(feature_cols) != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} feature columns plus label; found {len(feature_cols)}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    X = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = frame["label"].astype(str).str.upper().str.strip()
    if X.isna().any().any() or not X.apply(lambda c: c.map(lambda v: abs(v) != float('inf'))).all().all():
        raise ValueError("Feature data contains non-finite values")
    invalid = sorted(set(y) - set(LETTERS))
    if invalid:
        raise ValueError(f"Invalid labels: {invalid}. Labels must be A-Z.")
    if y.nunique() < 2:
        raise ValueError("At least two distinct labels are required for training")
    LOGGER.info("Loaded %d samples across %d classes", len(frame), y.nunique())
    return X, y

def stratified_split(X, y, test_size=0.2, random_state=42):
    counts = y.value_counts()
    if counts.min() < 2:
        raise ValueError("Each label needs at least two samples for a stratified split")
    test_count = math.ceil(len(y) * test_size) if isinstance(test_size, float) else test_size
    if test_count < y.nunique() or len(y) - test_count < y.nunique():
        raise ValueError("Dataset is too small for a stratified train/test split; add samples per class")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
