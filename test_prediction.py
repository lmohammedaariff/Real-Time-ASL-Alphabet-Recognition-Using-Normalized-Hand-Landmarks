from src.inference.smoothing import PredictionSmoother

def test_smoothing_requires_stable_votes():
    s = PredictionSmoother(5, .7, 3)
    assert s.update("A", .95) is None
    assert s.update("B", .8) is None
    assert s.update("A", .9) is None
    assert s.update("A", .9)[0] == "A"

def test_low_confidence_resets():
    s = PredictionSmoother(5, .7, 2)
    s.update("A", .9); assert s.update("A", .5) is None
