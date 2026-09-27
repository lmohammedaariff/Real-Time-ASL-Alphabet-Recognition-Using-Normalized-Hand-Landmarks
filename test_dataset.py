import pandas as pd
import pytest
from src.data.dataset_loader import load_landmark_csv
from src.data.dataset_collector import append_sample

def test_dataset_validation(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame([[0.0]*63+["A"], [1.0]*63+["B"]], columns=[*[f"f{i}" for i in range(63)], "label"]).to_csv(path, index=False)
    X, y = load_landmark_csv(path)
    assert X.shape == (2, 63) and set(y) == {"A", "B"}

def test_collector_rejects_bad_label(tmp_path):
    with pytest.raises(ValueError): append_sample(tmp_path / "x.csv", [0]*63, "?")
