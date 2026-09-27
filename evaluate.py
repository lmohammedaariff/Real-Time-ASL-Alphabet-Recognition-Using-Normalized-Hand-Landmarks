"""Evaluation report and plots for a fitted estimator."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".matplotlib"))
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

def evaluate_model(name, model, X_test, y_test, labels, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    pred = model.predict(X_test)
    latency = (time.perf_counter() - start) / max(len(X_test), 1)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    metrics = {"model": name, "accuracy": accuracy_score(y_test, pred), "precision": p,
               "recall": r, "f1_score": f1, "training_time": 0.0, "inference_time": latency}
    class_ids = list(range(len(labels)))
    cm = confusion_matrix(y_test, pred, labels=class_ids)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"{name} confusion matrix")
    fig.tight_layout(); fig.savefig(report_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=160); plt.close(fig)
    return metrics, classification_report(y_test, pred, labels=class_ids,
                                           target_names=labels, zero_division=0)
