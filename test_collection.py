from pathlib import Path

import pytest

from src.data.import_image_dataset import find_labeled_images
from src.data.collect_webcam import collect


def test_find_images_only_in_a_to_z_class_folders(tmp_path):
    (tmp_path / "A").mkdir()
    (tmp_path / "Z").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "A" / "one.jpg").write_bytes(b"not decoded in discovery")
    (tmp_path / "Z" / "two.PNG").write_bytes(b"not decoded in discovery")
    (tmp_path / "notes" / "ignore.jpg").write_bytes(b"ignore")
    results = list(find_labeled_images(tmp_path))
    assert [(label, path.name) for label, path in results] == [("A", "one.jpg"), ("Z", "two.PNG")]


def test_image_discovery_requires_existing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(find_labeled_images(tmp_path / "missing"))


def test_webcam_collector_reports_missing_camera(tmp_path):
    with pytest.raises(RuntimeError, match="unavailable"):
        collect("A", target=1, interval=0.1, output=tmp_path / "landmarks.csv", camera=99)
