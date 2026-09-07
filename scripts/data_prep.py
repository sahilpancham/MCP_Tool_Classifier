#!/usr/bin/env python3

"""Data loading and the text representations the models consume.

Training, evaluation and predict.py all build text here, so the representation
cannot drift between them.
"""

import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

BASE_MODEL = "Qwen/Qwen2.5-1.5B"

# 1,024 covers 99.7% of records untruncated (mean 106 tokens, max 1,461).
MAX_LENGTH = 1024

BATCH_SIZE = 8
SEED = 42

# Order defines the trained label ids and every confusion-matrix axis.
LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# server_slug is excluded on purpose: 40% of training servers hold a single
# category, so it is a shortcut that would not survive on unseen servers.
TEXT_FIELDS = ["name", "description", "input_schema"]


def load_jsonl(path):
    """Read a JSONL file into a DataFrame, skipping blank lines."""

    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(
            json.loads(line)
            for line in f
            if line.strip()
        )


def field(value):
    """Normalise a possibly-null record field to a string."""

    return "" if pd.isna(value) else str(value)


def compact_schema(schema_str):
    """Render a serialized JSON schema as compact natural language."""

    # Nulls are not screened out before json.loads. Missing and unparseable
    # schemas both fall through to `except`, and the adapter was trained on that
    # exact text, so an early return would feed it a string it never saw.
    try:
        schema = json.loads(schema_str)

        lines = []

        if schema.get("description"):
            lines.append(
                f"Schema description: {schema['description']}"
            )

        properties = schema.get("properties", {})

        if properties:
            lines.append("Parameters:")

            required = set(schema.get("required", []))

            for name, spec in properties.items():
                param_type = spec.get("type", "unknown")

                status = (
                    "required"
                    if name in required
                    else "optional"
                )

                line = (
                    f"- {name}: "
                    f"{param_type} "
                    f"({status})"
                )

                if spec.get("description"):
                    line += (
                        f" — {spec['description']}"
                    )

                if spec.get("enum"):
                    line += (
                        f" — allowed values: "
                        f"{spec['enum']}"
                    )

                lines.append(line)

        else:
            lines.append("Parameters: none")

        return "\n".join(lines)

    except Exception as exc:
        return f"Schema unavailable: {exc}"


def build_slm_text(record):
    """Input representation for the SLM, in training and at inference."""

    # Interpolated raw, not through field(): a null description renders as
    # "None" and a null schema as "Schema unavailable: ...not NoneType". Those
    # look like artefacts but are what the adapter saw for the 16% of records
    # with no schema, and normalising them changes predictions on those records.
    return (
        f"Tool name: {record['name']}\n"
        f"Description: {record['description']}\n"
        f"{compact_schema(record['input_schema'])}"
    )


def build_baseline_text(record):
    """Input representation for the TF-IDF baseline."""

    # Raw schema, not the SLM's rendering: n-grams over JSON key and type names
    # are informative to a linear model, which gains nothing from prose.
    return (
        f"Tool name: {field(record['name']).strip()}\n"
        f"Description: {field(record['description']).strip()}\n"
        f"Input schema: {field(record['input_schema']).strip()}"
    )


def require_columns(df, columns):
    """Fail fast on an input file that is missing fields we need."""

    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
