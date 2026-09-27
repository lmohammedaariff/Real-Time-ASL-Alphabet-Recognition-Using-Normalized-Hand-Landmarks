import av
import numpy as np

from src.inference.live_video import LivePredictionProcessor


def test_live_processor_returns_annotated_frame():
    processor = LivePredictionProcessor()
    try:
        source = av.VideoFrame.from_ndarray(np.zeros((240, 320, 3), dtype=np.uint8), format="bgr24")
        result = processor.recv(source)
        assert result.width == 320
        assert result.height == 240
        assert result.format.name == "bgr24"
    finally:
        processor.close()
