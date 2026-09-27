"""Central configuration for SignVision."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

@dataclass(frozen=True)
class Settings:
    camera_index: int = 0
    min_confidence: float = 0.70
    num_hands: int = 1
    prediction_window: int = 10
    stable_frames: int = 5
    accept_cooldown: float = 1.0
    target_samples: int = 500
    dataset_path: Path = ROOT / "data/landmarks/landmarks.csv"
    model_dir: Path = ROOT / "models"
    reports_dir: Path = ROOT / "reports"
    image_width: int = 960
    image_height: int = 720
    tts_enabled: bool = True

SETTINGS = Settings()
