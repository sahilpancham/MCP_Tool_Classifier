#!/usr/bin/env python3

"""Non-neural baseline: TF-IDF + Logistic Regression.

Writes the "baseline" section of metrics.json plus the confusion matrix and
pickled artifacts under outputs/baseline/. Trains on CPU in about 15 seconds.

Run from the project root:
    python -m scripts.train_baseline
"""

import pickle
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from scripts.data_prep import (
    DATA_DIR,
    OUTPUTS_DIR,
    SEED,
    build_baseline_text,
    load_jsonl,
)
from scripts.evaluation import (
    classification_metrics,
    save_confusion_matrix,
    update_metrics,
)


OUTPUT_DIR = OUTPUTS_DIR / "baseline"

BENCHMARK_EXAMPLES = 1000

TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 150000,
    "sublinear_tf": True,
}

CLASSIFIER_CONFIG = {
    "solver": "liblinear",
    # Inverse-frequency reweighting, the linear counterpart of the SLM's
    # weighted cross entropy.
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": SEED,
}


def megabytes(path):
    return path.stat().st_size / (1024 * 1024)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_jsonl(DATA_DIR / "train.jsonl")
    validation_df = load_jsonl(DATA_DIR / "validation.jsonl")

    print(f"Training records: {len(train_df)}")
    print(f"Validation records: {len(validation_df)}")

    train_text = train_df.apply(build_baseline_text, axis=1)
    validation_text = validation_df.apply(build_baseline_text, axis=1)

    y_train = train_df["category"]
    y_validation = validation_df["category"]

    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)

    # Fit on train only: train+validation would leak validation vocabulary and
    # document frequencies into the features.
    X_train = vectorizer.fit_transform(train_text)
    X_validation = vectorizer.transform(validation_text)

    print(f"TF-IDF train shape: {X_train.shape}")

    classifier = LogisticRegression(**CLASSIFIER_CONFIG)

    start = time.perf_counter()
    classifier.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    predictions = classifier.predict(X_validation)

    save_confusion_matrix(
        y_validation,
        predictions,
        OUTPUT_DIR,
        "Baseline validation confusion matrix",
    )

    model_path = OUTPUT_DIR / "baseline_model.pkl"
    vectorizer_path = OUTPUT_DIR / "tfidf_vectorizer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    # Time vectorisation and prediction together: that is what serving costs.
    benchmark_text = validation_text.iloc[:BENCHMARK_EXAMPLES]

    start = time.perf_counter()
    classifier.predict(vectorizer.transform(benchmark_text))
    elapsed = time.perf_counter() - start

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "features": ["name", "description", "input_schema"],
        "tfidf": {
            **TFIDF_CONFIG,
            "ngram_range": list(TFIDF_CONFIG["ngram_range"]),
        },
        "classifier": CLASSIFIER_CONFIG,
        "training_time_seconds": float(training_time),
        **classification_metrics(y_validation, predictions),
        "performance": {
            "benchmark_examples": int(len(benchmark_text)),
            "latency_per_example_ms": elapsed / len(benchmark_text) * 1000,
            "throughput_examples_per_second": len(benchmark_text) / elapsed,
            "model_pickle_size_mb": megabytes(model_path),
            "vectorizer_pickle_size_mb": megabytes(vectorizer_path),
            "total_artifact_size_mb": (
                megabytes(model_path) + megabytes(vectorizer_path)
            ),
        },
    }

    update_metrics("baseline", metrics)

    print("\nValidation results")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Training time: {training_time:.2f}s")
    print(f"\nSaved artifacts to: {OUTPUT_DIR} and metrics.json")


if __name__ == "__main__":
    main()
