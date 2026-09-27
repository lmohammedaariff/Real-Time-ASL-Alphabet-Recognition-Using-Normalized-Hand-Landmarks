import numpy as np
import pytest
from src.preprocessing.landmark_processor import normalize_landmarks

def test_normalize_shape_translation_and_scale_invariant():
    p = np.arange(63, dtype=np.float32).reshape(21, 3)
    a = normalize_landmarks(p)
    b = normalize_landmarks(p * 4 + 13)
    assert a.shape == (63,)
    np.testing.assert_allclose(a, b, atol=1e-6)

def test_invalid_landmarks_rejected():
    with pytest.raises(ValueError): normalize_landmarks(np.zeros((20, 3)))
    with pytest.raises(ValueError): normalize_landmarks(np.zeros((21, 3)))
