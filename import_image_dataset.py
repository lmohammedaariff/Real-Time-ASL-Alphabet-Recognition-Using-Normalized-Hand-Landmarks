"""Convert a labeled A-Z image-folder dataset into normalized hand landmarks.

Expected layout: ``dataset_root/A/*.jpg`` through ``dataset_root/Z/*.jpg``.
Nested dataset splits such as ``train/A`` and ``test/A`` are supported.
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from config import LETTERS, SETTINGS
from src.data.dataset_collector import append_feature_row
from src.preprocessing.landmark_processor import extract_hand_features

LOGGER = logging.getLogger("signvision.image_import")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_labeled_images(root: Path):
    """Yield (label, path) for images in directories named exactly A-Z."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    for folder in sorted(p for p in root.rglob("*") if p.is_dir() and p.name.upper() in LETTERS):
        label = folder.name.upper()
        for image in sorted(folder.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                yield label, image


def import_dataset(root: Path, output: Path, limit_per_class: int | None = None) -> dict:
    """Detect hands in labeled images and append usable landmark samples."""
    import cv2
    import mediapipe as mp

    root = Path(root)
    candidates = list(find_labeled_images(root))
    if not candidates:
        raise ValueError(f"No A-Z class folders containing supported images found under {root}")

    imported: Counter = Counter()
    skipped: Counter = Counter()
    hands_api = mp.solutions.hands
    with hands_api.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.5) as hands:
        for label, image_path in candidates:
            if limit_per_class is not None and imported[label] >= limit_per_class:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                skipped[label] += 1
                continue
            result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not result.multi_hand_landmarks:
                skipped[label] += 1
                continue
            try:
                features = extract_hand_features(result.multi_hand_landmarks[0])
                append_feature_row(output, features, label)
                imported[label] += 1
            except ValueError:
                skipped[label] += 1

    return {
        "images_found": len(candidates),
        "samples_imported": sum(imported.values()),
        "images_skipped": sum(skipped.values()),
        "per_class_imported": dict(sorted(imported.items())),
        "per_class_skipped": dict(sorted(skipped.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="root folder containing A-Z subfolders")
    parser.add_argument("--output", type=Path, default=SETTINGS.dataset_path)
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="optional cap, useful for a quick import check")
    args = parser.parse_args()
    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error("--limit-per-class must be positive")
    try:
        result = import_dataset(args.input, args.output, args.limit_per_class)
        print(f"Imported {result['samples_imported']} of {result['images_found']} images; "
              f"skipped {result['images_skipped']}. Saved CSV: {args.output}")
        for label in LETTERS:
            count = result["per_class_imported"].get(label, 0)
            if count:
                print(f"  {label}: {count} landmark samples")
    except (ImportError, FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Dataset import could not continue: {exc}\n")


if __name__ == "__main__":
    main()
