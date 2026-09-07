#!/usr/bin/env python3

"""Classify MCP tools from a JSONL file into one action category each.

Outputs exactly two columns, record_id and category, in input order.

    python predict.py \
      --model-path outputs/slm/final_adapter \
      --input data/test_unlabeled.jsonl \
      --output predictions.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from scripts.data_prep import LABELS, build_slm_text, load_jsonl, require_columns
from scripts.inference import load_model, predict_labels


DEFAULT_MODEL_PATH = "outputs/slm/final_adapter"

REQUIRED_COLUMNS = ["record_id", "name", "description", "input_schema"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify MCP tools using Qwen2.5-1.5B + LoRA."
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the LoRA adapter directory.",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file containing MCP tool records.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = load_jsonl(Path(args.input))

    # server_slug is not required: the model never sees it.
    require_columns(df, REQUIRED_COLUMNS)

    model, tokenizer, device = load_model(args.model_path)

    print(f"Device: {device}")
    print(f"Records: {len(df)}")

    predictions, _ = predict_labels(
        model,
        tokenizer,
        device,
        [build_slm_text(row) for _, row in df.iterrows()],
    )

    output = pd.DataFrame(
        {
            "record_id": df["record_id"].values,
            "category": predictions,
        }
    )

    # Check the contract before writing, so a broken run fails loudly instead
    # of producing a file that only looks right.
    if len(output) != len(df):
        raise RuntimeError("Prediction count does not match input count.")

    if not output["record_id"].is_unique:
        raise ValueError("Duplicate record_id values detected.")

    invalid = int((~output["category"].isin(LABELS)).sum())

    if invalid:
        raise ValueError(f"Invalid predictions detected: {invalid}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Rows: {len(output)}")
    print("Invalid predictions: 0")


if __name__ == "__main__":
    main()
