"""Download the public ASL hand-photo dataset and extract landmark features.

Images are streamed from the public Hugging Face dataset parquet into memory;
only the resulting landmark vectors and labels are stored in SignVision's CSV.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from urllib.parse import quote

import requests

from config import LETTERS, ROOT, SETTINGS
from src.data.dataset_collector import append_feature_row
from src.preprocessing.landmark_processor import extract_hand_features

DATASET_ID = "Marxulia/asl_sign_languages_alphabets_v03"
DATASET_API = "https://datasets-server.huggingface.co/parquet"
LOGGER = logging.getLogger("signvision.huggingface_import")


def fetch_parquet(dataset_id: str, cache_path: Path) -> Path:
    """Resolve and download the dataset's public parquet split to local cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    response = requests.get(DATASET_API, params={"dataset": dataset_id}, timeout=30)
    response.raise_for_status()
    manifests = response.json().get("parquet_files", [])
    if not manifests:
        raise RuntimeError(f"No public parquet split was returned for {dataset_id}")
    manifest = next((entry for entry in manifests if entry.get("split") == "train"), manifests[0])
    url = manifest["url"]
    temp_path = cache_path.with_suffix(cache_path.suffix + ".partial")
    with requests.get(url, stream=True, timeout=(30, 180)) as download:
        download.raise_for_status()
        with temp_path.open("wb") as stream:
            for chunk in download.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    temp_path.replace(cache_path)
    return cache_path


def extract_features(parquet_path: Path, output_csv: Path, max_images: int | None = None,
                     progress_every: int = 500) -> dict:
    """Extract hands from parquet image rows and append valid A-Z landmarks."""
    import cv2
    import mediapipe as mp
    import numpy as np
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    total = parquet.metadata.num_rows
    limit = min(total, max_images) if max_images is not None else total
    if limit < 1:
        raise ValueError("No image rows selected for extraction")
    imported = {letter: 0 for letter in LETTERS}
    skipped = 0
    processed = 0
    hands_api = mp.solutions.hands
    with hands_api.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.55) as hands:
        for batch in parquet.iter_batches(batch_size=64, columns=["image", "label"]):
            for row in batch.to_pylist():
                if processed >= limit:
                    break
                processed += 1
                label_id = row.get("label")
                image_field = row.get("image") or {}
                encoded = image_field.get("bytes") if isinstance(image_field, dict) else None
                if not isinstance(label_id, int) or not 0 <= label_id < len(LETTERS) or not encoded:
                    skipped += 1
                    continue
                image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    skipped += 1
                    continue
                result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not result.multi_hand_landmarks:
                    skipped += 1
                    continue
                try:
                    label = LETTERS[label_id]
                    append_feature_row(output_csv, extract_hand_features(result.multi_hand_landmarks[0]), label)
                    imported[label] += 1
                except ValueError as exc:
                    LOGGER.info("Skipping invalid sample %d: %s", processed, exc)
                    skipped += 1
                if processed % progress_every == 0:
                    LOGGER.info("Processed %d/%d images; extracted %d landmarks", processed, limit,
                                sum(imported.values()))
            if processed >= limit:
                break

    result = {
        "dataset_id": DATASET_ID,
        "dataset_card": f"https://huggingface.co/datasets/{quote(DATASET_ID, safe='/')}",
        "license_note": "The dataset card did not state an explicit image license; retain this provenance and verify permissions before redistribution or publication.",
        "images_processed": processed,
        "landmarks_extracted": sum(imported.values()),
        "images_skipped": skipped,
        "per_class": {key: value for key, value in imported.items() if value},
        "feature_csv": str(output_csv),
    }
    report_path = SETTINGS.reports_dir / "dataset_extraction.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SETTINGS.dataset_path)
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "data/raw/asl_alphabet_public.parquet")
    parser.add_argument("--max-images", type=int, default=None,
                        help="optional limit for a quick check; omit to process the full split")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    try:
        parquet_path = fetch_parquet(DATASET_ID, args.cache)
        result = extract_features(parquet_path, args.output, args.max_images)
        print(f"Extracted {result['landmarks_extracted']} landmark samples from "
              f"{result['images_processed']} images; skipped {result['images_skipped']}.")
        print(f"Feature data: {args.output}")
        print("Review reports/dataset_extraction.json and the dataset card/licensing before redistribution.")
    except (OSError, requests.RequestException, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Public dataset import failed: {exc}\n")


if __name__ == "__main__":
    main()
