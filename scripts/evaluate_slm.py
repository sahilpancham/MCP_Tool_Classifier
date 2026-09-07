#!/usr/bin/env python3

"""Score the fine-tuned SLM on validation and analyse its errors.

Writes the "slm" section of metrics.json plus the confusion matrix, every
misclassified record and the failure-mode report under outputs/slm/. Uses the
validation split only; the held-out test set is never scored here.

Run from the project root:
    python -m scripts.evaluate_slm
"""

import argparse
import json
import time
from pathlib import Path

import torch

from scripts.data_prep import (
    BASE_MODEL,
    BATCH_SIZE,
    DATA_DIR,
    LABELS,
    MAX_LENGTH,
    OUTPUTS_DIR,
    build_slm_text,
    field,
    load_jsonl,
)
from scripts.evaluation import (
    METRICS_PATH,
    classification_metrics,
    save_confusion_matrix,
    update_metrics,
)
from scripts.inference import load_model, predict_labels


OUTPUT_DIR = OUTPUTS_DIR / "slm"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "final_adapter"

# Enough batches to average out warm-up noise without re-running the split.
BENCHMARK_BATCHES = 10

# Per class rather than globally, so the minority classes are always covered.
ERRORS_PER_CLASS = 4

ERROR_COLUMNS = [
    "record_id",
    "server_slug",
    "name",
    "actual_category",
    "predicted_category",
    "confidence",
    "description",
    "input_schema",
]


def benchmark(model, tokenizer, device, model_path, texts):
    """Measure serving cost: latency, throughput, memory and artifact size."""

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sample = texts[:BATCH_SIZE * BENCHMARK_BATCHES]

    start = time.perf_counter()

    predict_labels(model, tokenizer, device, sample, show_progress=False)

    elapsed = time.perf_counter() - start

    parameters = list(model.parameters())

    trainable = [p for p in parameters if p.requires_grad]
    frozen = [p for p in parameters if not p.requires_grad]

    total_parameters = sum(p.numel() for p in parameters)
    trainable_parameters = sum(p.numel() for p in trainable)

    performance = {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percentage": round(
            trainable_parameters / total_parameters * 100, 4
        ),
        "adapter_size_mb": round(
            sum(
                item.stat().st_size
                for item in model_path.glob("adapter_model.*")
            )
            / (1024 * 1024),
            2,
        ),
        # Frozen only: the base weights a deployment hosts alongside the adapter.
        "base_model_size_gb": round(
            sum(p.numel() * p.element_size() for p in frozen) / 1e9, 2
        ),
        "benchmark_batch_size": BATCH_SIZE,
        "max_sequence_length": MAX_LENGTH,
        "batch_latency_seconds": elapsed / BENCHMARK_BATCHES,
        "latency_per_example_ms": elapsed / len(sample) * 1000,
        "throughput_examples_per_second": len(sample) / elapsed,
    }

    if device.type == "cuda":
        performance["peak_gpu_memory_gb"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )

    return performance


def representative_errors(errors):
    """Highest-confidence errors per actual class.

    Cases the model was most sure about and still wrong show what it learned,
    rather than where it was merely undecided.
    """

    selected = []

    for label in LABELS:
        subset = (
            errors[errors["actual_category"] == label]
            .sort_values("confidence", ascending=False)
            .head(ERRORS_PER_CLASS)
        )

        for record in subset.to_dict("records"):
            # to_dict yields numpy scalars and NaN, which json.dump cannot write.
            clean = {key: field(value) for key, value in record.items()}
            clean["confidence"] = float(record["confidence"])

            selected.append(clean)

    return selected


def failure_modes(errors, matrix):
    """The three recurring confusions, counted from the confusion matrix."""

    position = {label: i for i, label in enumerate(LABELS)}

    def confused(actual, predicted):
        return int(matrix[position[actual], position[predicted]])

    return [
        {
            "name": "Read vs Write ambiguity",
            "evidence": {
                "Write_to_Read": confused("Write", "Read"),
                "Read_to_Write": confused("Read", "Write"),
                "combined": (
                    confused("Write", "Read") + confused("Read", "Write")
                ),
            },
            "description": (
                "Information-returning tools and state-changing tools are "
                "swapped. The model keys on wording such as 'returns' or "
                "'generate' rather than on whether state changes."
            ),
        },
        {
            "name": "Execute vs Read/Write ambiguity",
            "evidence": {
                "Execute_to_Read": confused("Execute", "Read"),
                "Execute_to_Write": confused("Execute", "Write"),
                "Read_to_Execute": confused("Read", "Execute"),
                "Write_to_Execute": confused("Write", "Execute"),
            },
            "description": (
                "Workflows, browser actions and analysis tools are labelled "
                "by surface wording rather than by whether they run an "
                "operation. Execute is the vaguest category and several of "
                "these are genuinely ambiguous."
            ),
        },
        {
            "name": "Minority-class collapse",
            "evidence": {
                f"{label}_errors": int(
                    (errors["actual_category"] == label).sum()
                )
                for label in ["Other", "Financial", "Destructive"]
            },
            "description": (
                "Other is never predicted at all: 76 training examples and no "
                "shared surface signal, which class weighting alone did not "
                "fix. Financial and Destructive are absorbed into Write."
            ),
        },
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the LoRA-adapted SLM on the validation split."
    )

    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the LoRA adapter directory.",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_df = load_jsonl(DATA_DIR / "validation.jsonl")

    model, tokenizer, device = load_model(model_path)

    print(f"Device: {device}")
    print(f"Validation records: {len(validation_df)}")

    texts = [build_slm_text(row) for _, row in validation_df.iterrows()]

    predictions, confidences = predict_labels(model, tokenizer, device, texts)

    results = validation_df.assign(
        predicted_category=predictions,
        confidence=confidences,
    ).rename(columns={"category": "actual_category"})

    matrix = save_confusion_matrix(
        results["actual_category"],
        results["predicted_category"],
        OUTPUT_DIR,
        "SLM validation confusion matrix",
    )

    errors = results[
        results["actual_category"] != results["predicted_category"]
    ][ERROR_COLUMNS]

    errors.to_csv(OUTPUT_DIR / "validation_errors.csv", index=False)

    error_analysis = {
        "validation_error_count": int(len(errors)),
        "validation_error_rate": float(len(errors) / len(validation_df)),
        "errors_by_actual_class": {
            label: int((errors["actual_category"] == label).sum())
            for label in LABELS
        },
        "recurring_failure_modes": failure_modes(errors, matrix),
        "example_selection": (
            f"Up to {ERRORS_PER_CLASS} highest-confidence errors per actual "
            "class, so every class including the minority classes is covered."
        ),
        "examples": representative_errors(errors),
    }

    with open(
        OUTPUT_DIR / "error_analysis.json", "w", encoding="utf-8"
    ) as f:
        json.dump(error_analysis, f, indent=2)

    metrics = {
        "model": f"{BASE_MODEL} + LoRA",
        # Structurally zero, since prediction is an argmax over six logits.
        # Computed rather than asserted.
        "invalid_output_rate": float(
            sum(1 for label in predictions if label not in LABELS)
            / len(predictions)
        ),
        "labels": LABELS,
        **classification_metrics(
            results["actual_category"], results["predicted_category"]
        ),
        "performance": benchmark(
            model, tokenizer, device, model_path, texts
        ),
    }

    update_metrics("slm", metrics)

    print("\nValidation results")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Errors:      {len(errors)}")
    print(f"\nSaved artifacts to: {OUTPUT_DIR} and {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
