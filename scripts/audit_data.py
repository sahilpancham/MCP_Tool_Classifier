#!/usr/bin/env python3

"""Data audit: distribution, missing fields, duplicates and leakage risk.

Writes outputs/audit/data_audit.json. Run from the project root:
    python -m scripts.audit_data
"""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from scripts.data_prep import (
    DATA_DIR,
    LABELS,
    OUTPUTS_DIR,
    TEXT_FIELDS,
    field,
    load_jsonl,
)


OUTPUT_DIR = OUTPUTS_DIR / "audit"

NEAR_DUPLICATE_THRESHOLD = 0.9
NEAR_DUPLICATE_MAX_FEATURES = 100000

# Terms unique to one tool cannot make two tools look alike.
NEAR_DUPLICATE_MIN_DF = 2

# Similarities are computed in row blocks: the full 22k x 22k dense matrix
# would need ~4 GB, while one block of 512 rows needs ~90 MB.
NEAR_DUPLICATE_CHUNK = 512

NEAR_DUPLICATE_EXAMPLES = 20

SPLIT_PAIRS = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

# Every count is ambiguous without its rule, so the rules ship with the numbers.
DEFINITIONS = {
    "missing_values": "Count of null values per column.",
    "empty_text_fields": "Count of values that are null or whitespace-only.",
    "schema_validity": (
        "input_schema parsed with json.loads: valid_object is a JSON object, "
        "non_object is valid JSON that is not an object, malformed fails to parse."
    ),
    "exact_duplicates": (
        "SHA-256 over name + description + input_schema. within_split counts "
        "rows after the first in each content group; cross_split counts content "
        "groups present in both splits."
    ),
    "literal_category_occurrences": (
        "Training rows where the category word appears as a case-insensitive "
        "substring of the field."
    ),
    "exact_category_tool_names": (
        "Training rows whose lowercased tool name is exactly the category word."
    ),
    "near_duplicates": (
        "Training rows whose nearest *other* training row, by cosine "
        "similarity over TF-IDF of the tool text, scores at least "
        f"{NEAR_DUPLICATE_THRESHOLD}."
    ),
}


def content_hash(record):
    """Identity of a tool's content, ignoring record_id and server_slug."""

    text = "\n".join(field(record[column]) for column in TEXT_FIELDS)

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def schema_status(value):
    if pd.isna(value) or str(value).strip() == "":
        return "missing"

    try:
        parsed = json.loads(value)
    except Exception:
        return "malformed"

    return "valid_object" if isinstance(parsed, dict) else "non_object"


def audited_columns(df):
    columns = ["record_id", "server_slug", *TEXT_FIELDS]

    if "category" in df.columns:
        columns.append("category")

    return columns


def near_duplicate_analysis(train):
    """Find training rows that are near-copies of another training row.

    Server-isolated splits do not prevent content-level duplication.
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=NEAR_DUPLICATE_MIN_DF,
        max_features=NEAR_DUPLICATE_MAX_FEATURES,
        sublinear_tf=True,
    )

    # The field labels used for model input are omitted here: they are identical
    # in every row, so they would only add a constant similarity floor.
    text = (
        train["name"].map(field).str.strip()
        + "\n"
        + train["description"].map(field).str.strip()
        + "\n"
        + train["input_schema"].map(field).str.strip()
    )

    # TF-IDF rows are L2-normalised by default, so a linear kernel over them
    # is exactly cosine similarity.
    matrix = vectorizer.fit_transform(text)

    row_count = matrix.shape[0]

    best_similarity = np.zeros(row_count)
    best_index = np.zeros(row_count, dtype=int)

    for start in range(0, row_count, NEAR_DUPLICATE_CHUNK):
        stop = min(start + NEAR_DUPLICATE_CHUNK, row_count)

        similarities = linear_kernel(matrix[start:stop], matrix)

        # Every row is its own nearest neighbour at similarity 1.0, so blank
        # out the diagonal before taking the maximum.
        for offset in range(stop - start):
            similarities[offset, start + offset] = -1.0

        best_index[start:stop] = similarities.argmax(axis=1)
        best_similarity[start:stop] = similarities.max(axis=1)

    is_near_duplicate = best_similarity >= NEAR_DUPLICATE_THRESHOLD

    ranked = np.argsort(-best_similarity)[:NEAR_DUPLICATE_EXAMPLES]

    examples = []

    for i in ranked:
        a = train.iloc[int(i)]
        b = train.iloc[int(best_index[i])]

        examples.append(
            {
                "similarity": float(best_similarity[i]),
                "record_id_a": a["record_id"],
                "record_id_b": b["record_id"],
                "name_a": a["name"],
                "name_b": b["name"],
                "label_a": a["category"],
                "label_b": b["category"],
            }
        )

    return {
        "method": {
            "representation": "TF-IDF over name + description + input_schema",
            "ngram_range": [1, 2],
            "min_df": NEAR_DUPLICATE_MIN_DF,
            "max_features": NEAR_DUPLICATE_MAX_FEATURES,
            "sublinear_tf": True,
            "similarity_metric": "cosine similarity",
            "threshold": NEAR_DUPLICATE_THRESHOLD,
        },
        "near_duplicate_records": int(is_near_duplicate.sum()),
        "percentage_of_training": round(
            float(is_near_duplicate.mean() * 100), 4
        ),
        "note": (
            "Near-duplicates are retained. This is an audit finding; the "
            "supplied split is not modified."
        ),
        "examples": examples,
    }


def target_leakage_analysis(train):
    """Check whether the label can be read off the input.

    Literal category words are not leakage, since action verbs occur naturally.
    Category-pure servers are the real risk.
    """

    lowered = {
        column: train[column].map(field).str.lower()
        for column in ["name", "description", "input_schema", "server_slug"]
    }

    categories_per_server = train.groupby("server_slug")["category"].nunique()

    single_category_servers = int((categories_per_server == 1).sum())
    total_servers = int(len(categories_per_server))

    concentration = single_category_servers / total_servers * 100

    return {
        "literal_category_occurrences": {
            column: {
                label: int(
                    series.str.contains(label.lower(), regex=False).sum()
                )
                for label in LABELS
            }
            for column, series in lowered.items()
        },
        "exact_category_tool_names": {
            label: int(lowered["name"].eq(label.lower()).sum())
            for label in LABELS
        },
        "server_label_concentration": {
            "single_category_servers": single_category_servers,
            "total_servers": total_servers,
            "percentage": round(concentration, 4),
        },
        "interpretation": [
            "Category words occur literally in some input fields, which is not "
            "by itself proof of leakage.",
            f"{concentration:.2f}% of training servers contain exactly one "
            "category, so server identity is a strong shortcut.",
            "server_slug is therefore excluded from model input.",
            "Server-level split isolation prevents training-server identities "
            "from reappearing in validation or test.",
        ],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": load_jsonl(DATA_DIR / "train.jsonl"),
        "validation": load_jsonl(DATA_DIR / "validation.jsonl"),
        "test": load_jsonl(DATA_DIR / "test_unlabeled.jsonl"),
    }

    train = splits["train"]

    hashes = {
        name: df.apply(content_hash, axis=1)
        for name, df in splits.items()
    }

    servers = {
        name: set(df["server_slug"])
        for name, df in splits.items()
    }

    label_counts = train["category"].value_counts()

    # Label sets per content group, to find identical tools labelled differently.
    train_labels_by_content = train.groupby(hashes["train"])["category"].agg(set)

    validation_labels_by_content = (
        splits["validation"].groupby(hashes["validation"])["category"].agg(set)
    )

    shared_content = train_labels_by_content.index.intersection(
        validation_labels_by_content.index
    )

    conflicting_content = sum(
        1
        for key in shared_content
        if train_labels_by_content[key] != validation_labels_by_content[key]
    )

    audit = {
        "definitions": DEFINITIONS,
        "dataset_sizes": {
            name: int(len(df))
            for name, df in splits.items()
        },
        "missing_values": {
            name: {
                column: int(df[column].isna().sum())
                for column in audited_columns(df)
            }
            for name, df in splits.items()
        },
        "duplicate_record_ids": {
            name: int(df["record_id"].duplicated().sum())
            for name, df in splits.items()
        },
        "empty_text_fields": {
            name: {
                column: int(df[column].map(field).str.strip().eq("").sum())
                for column in TEXT_FIELDS
            }
            for name, df in splits.items()
        },
        "schema_validity": {
            name: {
                status: int(
                    (df["input_schema"].map(schema_status) == status).sum()
                )
                for status in [
                    "valid_object",
                    "missing",
                    "malformed",
                    "non_object",
                ]
            }
            for name, df in splits.items()
        },
        "train_label_distribution": {
            label: {
                "count": int(label_counts.get(label, 0)),
                "percentage": round(
                    float(label_counts.get(label, 0) / len(train) * 100), 4
                ),
            }
            for label in LABELS
        },
        "server_isolation": {
            "unique_servers": {
                name: len(value)
                for name, value in servers.items()
            },
            "overlap": {
                f"{a}_{b}": len(servers[a] & servers[b])
                for a, b in SPLIT_PAIRS
            },
            "passed": all(
                not servers[a] & servers[b]
                for a, b in SPLIT_PAIRS
            ),
        },
        "exact_duplicates": {
            "within_split": {
                name: int(series.duplicated().sum())
                for name, series in hashes.items()
            },
            "cross_split": {
                f"{a}_{b}": len(set(hashes[a]) & set(hashes[b]))
                for a, b in SPLIT_PAIRS
            },
        },
        "cross_split_label_consistency": {
            "train_validation_shared_content_groups": int(len(shared_content)),
            "train_validation_conflicting_groups": int(conflicting_content),
            "test": "Not evaluable because test labels are unavailable.",
        },
        "target_leakage": target_leakage_analysis(train),
        "near_duplicates": near_duplicate_analysis(train),
    }

    with open(OUTPUT_DIR / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit["dataset_sizes"], indent=2))
    print(f"Server isolation passed: {audit['server_isolation']['passed']}")
    print(
        "Near-duplicate training records: "
        f"{audit['near_duplicates']['near_duplicate_records']} "
        f"({audit['near_duplicates']['percentage_of_training']}%)"
    )
    print(f"\nSaved audit to: {OUTPUT_DIR / 'data_audit.json'}")


if __name__ == "__main__":
    main()
