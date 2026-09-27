"""Small shared helpers."""
from pathlib import Path

def ensure_project_dirs(root: Path):
    for rel in ("data/raw", "data/processed", "data/landmarks", "models", "reports", "logs"):
        (root / rel).mkdir(parents=True, exist_ok=True)
