#!/usr/bin/env python3

"""Model loading and batched prediction, shared by predict.py and evaluation.

Framed as sequence classification, so a prediction is one argmax over six
logits and an invalid label is not representable.
"""

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.data_prep import (
    BASE_MODEL,
    BATCH_SIZE,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    MAX_LENGTH,
)


def load_model(model_path):
    """Load the base model plus the LoRA adapter at `model_path`."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The tokenizer is loaded from the adapter directory rather than from the
    # Hub, so inference uses exactly the vocabulary training used.
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        torch_dtype=(
            torch.float16 if device.type == "cuda" else torch.float32
        ),
    )

    model = PeftModel.from_pretrained(base_model, model_path)

    # A decoder-only classifier pools the last non-padding token, so it cannot
    # find the end of a sequence without pad_token_id. Set on both objects
    # because PeftModel carries its own config.
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_labels(model, tokenizer, device, texts, show_progress=True):
    """Classify `texts` in batches, returning (labels, confidences).

    Confidence is the max softmax probability, used for error analysis only.
    It is not calibrated.
    """

    labels = []
    confidences = []

    starts = range(0, len(texts), BATCH_SIZE)

    for start in tqdm(starts, desc="Inference", disable=not show_progress):
        encoded = tokenizer(
            texts[start:start + BATCH_SIZE],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )

        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits

        # Cast to fp32 before softmax: fp16 probabilities are too coarse to be
        # useful as confidences.
        probabilities = torch.softmax(logits.float(), dim=-1)

        batch_confidences, batch_ids = probabilities.max(dim=-1)

        labels.extend(ID2LABEL[int(x)] for x in batch_ids.cpu().numpy())
        confidences.extend(float(x) for x in batch_confidences.cpu().numpy())

    return labels, confidences
