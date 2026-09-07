#!/usr/bin/env python3

"""Fine-tune Qwen2.5-1.5B with LoRA for MCP tool classification.

Writes the "training" section of metrics.json and the selected adapter to
outputs/slm/final_adapter/. Takes about two hours on a single 16 GB T4.

Run from the project root:
    python -m scripts.train_slm
"""

from pathlib import Path

import torch
import torch.nn as nn

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from scripts.data_prep import (
    BASE_MODEL,
    DATA_DIR,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    MAX_LENGTH,
    OUTPUTS_DIR,
    SEED,
    build_slm_text,
    load_jsonl,
)
from scripts.evaluation import update_metrics


OUTPUT_DIR = OUTPUTS_DIR / "slm"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FINAL_ADAPTER_DIR = OUTPUT_DIR / "final_adapter"

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # The head starts randomly initialised, so it trains and saves with the adapter.
    "modules_to_save": ["score"],
}

TRAINING_CONFIG = {
    "num_train_epochs": 2,
    # 2 is what fits in 16 GB at length 1,024; accumulation restores a batch of 16.
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,
}


def prepare_dataset(df):
    return Dataset.from_list(
        [
            {
                "text": build_slm_text(row),
                "labels": LABEL2ID[row["category"]],
            }
            for _, row in df.iterrows()
        ]
    )


def compute_metrics(eval_prediction):
    """Per-epoch validation scores; macro F1 drives checkpoint selection."""

    predictions, labels = eval_prediction

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predicted_ids = predictions.argmax(axis=-1)

    label_ids = list(range(len(LABELS)))

    return {
        "accuracy": float(accuracy_score(labels, predicted_ids)),
        "macro_f1": float(
            f1_score(
                labels,
                predicted_ids,
                average="macro",
                labels=label_ids,
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predicted_ids,
                average="weighted",
                labels=label_ids,
                zero_division=0,
            )
        ),
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross entropy.

    Read outnumbers Other 185:1; unweighted training collapses onto Read.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(
                device=logits.device,
                dtype=logits.dtype,
            )
        )

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_jsonl(DATA_DIR / "train.jsonl")
    validation_df = load_jsonl(DATA_DIR / "validation.jsonl")

    print(f"Training records: {len(train_df)}")
    print(f"Validation records: {len(validation_df)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        # Pad in the collator, so each batch pads to its own longest sequence.
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    train_dataset = prepare_dataset(train_df).map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    validation_dataset = prepare_dataset(validation_df).map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    # weight(class) = N / (K * count(class))
    class_counts = train_df["category"].value_counts()

    class_weights = torch.tensor(
        [
            len(train_df) / (len(LABELS) * class_counts[label])
            for label in LABELS
        ],
        dtype=torch.float32,
    )

    print("Class weights:", dict(zip(LABELS, class_weights.tolist())))

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        torch_dtype=(
            torch.float16 if torch.cuda.is_available() else torch.float32
        ),
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    model = get_peft_model(
        model,
        LoraConfig(task_type=TaskType.SEQ_CLS, **LORA_CONFIG),
    )

    # PEFT leaves new parameters in the base dtype, and FP16 trainable
    # parameters make the gradient scaler emit inf/NaN and stall at step one.
    # The frozen base weights stay FP16, where the memory saving comes from.
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.config.use_cache = False

    # Recompute activations instead of storing them, to fit 16 GB. Checkpointing
    # needs one input requiring grad, and every base weight here is frozen.
    model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        **TRAINING_CONFIG,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        ),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    # load_best_model_at_end restored the highest-macro-F1 checkpoint, so this
    # saves the selected adapter rather than the last epoch's.
    trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    evaluations = [
        {
            key: entry[key]
            for key in [
                "epoch",
                "step",
                "eval_loss",
                "eval_accuracy",
                "eval_macro_f1",
                "eval_weighted_f1",
            ]
            if key in entry
        }
        for entry in trainer.state.log_history
        if "eval_macro_f1" in entry
    ]

    best_evaluation = max(evaluations, key=lambda entry: entry["eval_macro_f1"])

    checkpoint = trainer.state.best_model_checkpoint

    update_metrics(
        "training",
        {
            "base_model": BASE_MODEL,
            "lora": LORA_CONFIG,
            "training_args": TRAINING_CONFIG,
            "effective_batch_size": (
                TRAINING_CONFIG["per_device_train_batch_size"]
                * TRAINING_CONFIG["gradient_accumulation_steps"]
            ),
            "max_sequence_length": MAX_LENGTH,
            "seed": SEED,
            "class_weights": dict(zip(LABELS, class_weights.tolist())),
            "checkpoint_selection": "highest validation macro F1",
            "selected_checkpoint": (
                None if checkpoint is None else Path(checkpoint).name
            ),
            "selected_epoch": best_evaluation["epoch"],
            "best_macro_f1": trainer.state.best_metric,
            "evaluations": evaluations,
        },
    )

    print("\nTraining completed.")
    print(f"Selected checkpoint: {checkpoint}")
    print(f"Best validation macro F1: {trainer.state.best_metric}")
    print(f"Saved adapter to: {FINAL_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
