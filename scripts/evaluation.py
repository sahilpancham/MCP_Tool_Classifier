#!/usr/bin/env python3

"""Scoring, confusion-matrix rendering, and the metrics.json deliverable.

The baseline and the SLM score through the same functions so their numbers are
comparable, and each owns one top-level section of metrics.json.
"""

import json

import matplotlib

# Select a non-interactive backend before pyplot is imported, so the scripts
# run on a headless GPU box.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from scripts.data_prep import LABELS, PROJECT_DIR  # noqa: E402


METRICS_PATH = PROJECT_DIR / "metrics.json"


def classification_metrics(y_true, y_pred):
    """Accuracy, macro F1, weighted F1 and per-class precision/recall/F1."""

    # labels=LABELS pins the class set, so a never-predicted class scores zero
    # instead of dropping out of the macro average and hiding the Other collapse.
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
    }


def save_confusion_matrix(y_true, y_pred, output_dir, title):
    """Write the confusion matrix as a CSV table and a PNG, returning the counts."""

    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)

    pd.DataFrame(
        matrix,
        index=[f"Actual_{label}" for label in LABELS],
        columns=[f"Predicted_{label}" for label in LABELS],
    ).to_csv(output_dir / "confusion_matrix.csv")

    # Shade by per-row rate, not raw count: with 2,817 Read and 15 Other
    # examples, a count-shaded image would be a single dark cell.
    row_totals = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    shading = matrix / row_totals

    figure, axes = plt.subplots(figsize=(7, 6))

    axes.imshow(shading, cmap="Blues", vmin=0, vmax=1)

    axes.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    axes.set_yticks(range(len(LABELS)), LABELS)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    axes.set_title(title)

    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            axes.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if shading[row, column] > 0.5 else "black",
            )

    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    return matrix


def update_metrics(section, payload):
    """Merge one section into metrics.json, leaving the others untouched.

    Lets the scripts run in any order and be re-run individually.
    """

    metrics = {
        "primary_metric": "macro_f1",
        "evaluation_split": "validation",
    }

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics.update(json.load(f))

    metrics[section] = payload

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
