"""Compare lightweight classifiers on a stratified landmark dataset."""
import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
import joblib
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from config import SETTINGS
from src.data.dataset_loader import load_landmark_csv, stratified_split
from src.training.evaluate import evaluate_model
from src.utils.logger import get_logger

def train(dataset=SETTINGS.dataset_path, model_dir=SETTINGS.model_dir, reports_dir=SETTINGS.reports_dir,
          random_state=42):
    logger = get_logger("signvision.training")
    X, y_text = load_landmark_csv(dataset)
    encoder = LabelEncoder().fit(y_text)
    y = pd.Series(encoder.transform(y_text), index=y_text.index)
    labels = list(encoder.classes_)
    X_train, X_test, y_train, y_test = stratified_split(X, y, random_state=random_state)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced")),
        "SVM": make_pipeline(StandardScaler(), SVC(C=4, probability=True, class_weight="balanced", random_state=random_state)),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=random_state),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                         validation_fraction=0.15, max_iter=500, random_state=random_state)),
    }
    model_dir, reports_dir = Path(model_dir), Path(reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True); reports_dir.mkdir(parents=True, exist_ok=True)
    rows, reports, fitted = [], [], {}
    for name, model in models.items():
        logger.info("Training %s", name)
        started = time.perf_counter(); model.fit(X_train, y_train); duration = time.perf_counter() - started
        metrics, report = evaluate_model(name, model, X_test, y_test, labels, reports_dir)
        metrics["training_time"] = duration
        rows.append(metrics); reports.append(f"## {name}\n{report}"); fitted[name] = model
    comparison = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    (reports_dir / "classification_report.txt").write_text("\n\n".join(reports), encoding="utf-8")
    for metric in ("accuracy", "f1_score"):
        ax = comparison.set_index("model")[metric].plot(kind="bar", color="#2b8a78", ylim=(0, 1), title=f"Model {metric}")
        ax.set_ylabel(metric.replace("_", " ")); ax.figure.tight_layout()
        ax.figure.savefig(reports_dir / f"{metric}_comparison.png", dpi=160); plt.close(ax.figure)
    best_name = str(comparison.iloc[0]["model"])
    best = fitted[best_name]
    best_matrix = reports_dir / f"confusion_matrix_{best_name.lower().replace(' ', '_')}.png"
    if best_matrix.exists():
        shutil.copy2(best_matrix, reports_dir / "confusion_matrix.png")
    joblib.dump({"model": best, "labels": labels, "model_name": best_name}, model_dir / "best_model.joblib")
    joblib.dump(encoder, model_dir / "label_encoder.joblib")
    for name, model in fitted.items():
        joblib.dump({"model": model, "labels": labels, "model_name": name}, model_dir / f"{name.lower().replace(' ', '_')}.joblib")
    metadata = {"best_model": best_name, "labels": labels, "samples": len(X), "features": X.shape[1],
                "random_state": random_state, "test_fraction": 0.2, "comparison": rows}
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Best model: %s", best_name)
    return comparison

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=SETTINGS.dataset_path)
    parser.add_argument("--models", type=Path, default=SETTINGS.model_dir)
    parser.add_argument("--reports", type=Path, default=SETTINGS.reports_dir)
    args = parser.parse_args()
    try:
        print(train(args.dataset, args.models, args.reports).to_string(index=False))
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Training could not start: {exc}")

if __name__ == "__main__":
    main()
