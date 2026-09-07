# MCP Tool Classification — AI Engineer Take-Home

Classify an MCP (Model Context Protocol) tool into exactly one action
category: `Read`, `Write`, `Execute`, `Destructive`, `Financial`, `Other`.

Two approaches were built and compared on the supplied server-isolated splits:

1. **Baseline** — TF-IDF + Logistic Regression
2. **SLM** — `Qwen/Qwen2.5-1.5B` adapted with LoRA (1.5B params, well inside the
   3B / 24 GB budget; trained on a single 16 GB T4)

Primary metric is **macro F1**, because the label distribution is extremely
skewed (63.6% `Read` versus 0.34% `Other`).

| Metric | Baseline | SLM | Δ |
|---|---:|---:|---:|
| **Macro F1** | 0.6738 | **0.7220** | +0.0481 |
| Accuracy | 0.8751 | **0.9352** | +6.01 pp |
| Weighted F1 | 0.8737 | **0.9332** | +0.0595 |

---

## 1. Setup

Requires Python 3.10+ and, for the SLM, a CUDA GPU with 16 GB VRAM or more.
The baseline and the data audit run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install a torch build matching your CUDA version first (see pytorch.org),
# then the rest:
pip install -r requirements.txt
```

Base model weights are **not** included. They download from the Hugging Face
Hub on first use (~3.1 GB). Only the ~8 MB LoRA adapter ships, in
`outputs/slm/final_adapter/`.

## 2. Commands

Run everything from the project root.

```bash
# Data audit -> outputs/audit/data_audit.json
python -m scripts.audit_data

# Baseline train, evaluate, benchmark -> outputs/baseline/, metrics.json
python -m scripts.train_baseline

# SLM fine-tuning -> outputs/slm/final_adapter/, metrics.json (~2 h on a T4)
python -m scripts.train_slm

# SLM scoring and error analysis -> outputs/slm/, metrics.json
python -m scripts.evaluate_slm

# Inference on the held-out test set -> predictions.csv
python predict.py \
  --model-path outputs/slm/final_adapter \
  --input data/test_unlabeled.jsonl \
  --output predictions.csv
```

`predict.py` checks its own output contract — row count, unique `record_id`,
valid labels — and raises rather than writing a malformed file. Output is
exactly:

```csv
record_id,category
```

Every file under `outputs/` and every number in `metrics.json` is produced by
these commands. Nothing was written by hand.

## 3. Repository layout

```
.
├── predict.py                    # inference entry point
├── requirements.txt
├── README.md
├── MODEL_CARD.md
├── metrics.json                  # all metrics: baseline, training, slm
├── predictions.csv               # test-set predictions
├── data/                         # supplied splits, unmodified
├── scripts/
│   ├── data_prep.py              # loading + input representation (shared)
│   ├── inference.py              # model loading + batched prediction (shared)
│   ├── evaluation.py             # scoring + confusion matrix + metrics.json (shared)
│   ├── audit_data.py             # data audit
│   ├── train_baseline.py         # baseline
│   ├── train_slm.py              # LoRA fine-tuning
│   └── evaluate_slm.py           # SLM scoring + error analysis
└── outputs/
    ├── audit/data_audit.json
    ├── baseline/                 # confusion matrix, fitted model
    └── slm/
        ├── final_adapter/        # the shipped model artifact
        ├── confusion_matrix.csv / .png
        ├── validation_errors.csv # all 287 errors
        └── error_analysis.json
```

Three design choices are worth calling out:

- **`scripts/data_prep.py` is the only definition of the model input.**
  Training, evaluation and `predict.py` all import `build_slm_text` from it, so
  the representation cannot drift between training and serving.
- **`scripts/inference.py` is the only prediction path.** Evaluation and
  `predict.py` run identical code, so validation numbers describe the thing
  that actually ships.
- **Each script owns one section of `metrics.json`** (`baseline`, `training`,
  `slm`), so no metric is stored in two places and scripts can be re-run
  individually.

---

## 4. Dataset

| Split | Records | Unique servers |
|---|---:|---:|
| Train | 22,143 | 1,448 |
| Validation | 4,429 | 292 |
| Test | 4,428 | 291 |

Splits are isolated by `server_slug` and were used exactly as given. No rows
were merged and no new row-level split was created. The audit verifies zero
`server_slug` overlap across all three pairs.

The two remaining supplied files, `servers_public.jsonl` and
`split_summary.json`, were not used and are not included. Server metadata is
deliberately kept out of model input (§5.4), and the split sizes above are
recomputed by the audit rather than read from the summary.

---

## 5. Data audit

Full report: `outputs/audit/data_audit.json`, which ships the counting rule for
every statistic alongside the numbers.

### 5.1 Label distribution

| Category | Count | Share |
|---|---:|---:|
| Read | 14,084 | 63.60% |
| Write | 5,433 | 24.54% |
| Destructive | 1,264 | 5.71% |
| Execute | 1,175 | 5.31% |
| Financial | 111 | 0.50% |
| Other | 76 | 0.34% |

`Read` outnumbers `Other` 185:1. This is why macro F1 is the primary metric and
why both models weight their loss by inverse class frequency.

### 5.2 Missing and malformed fields

| Split | Missing description | Missing input_schema | Malformed schema |
|---|---:|---:|---:|
| Train | 316 | 3,617 (16.3%) | 0 |
| Validation | 6 | 737 | 0 |
| Test | 42 | 743 | 0 |

No schema failed to parse. Missing schemas render as an explicit
`Schema unavailable: ...` line rather than an empty string, so the model can use
the absence as a signal instead of seeing a truncated prompt.

### 5.3 Duplicates and near-duplicates

Exact content is hashed over `name + description + input_schema`.

| Split | Duplicate rows | Unique contents |
|---|---:|---:|
| Train | 786 | 21,357 |
| Validation | 32 | 4,397 |
| Test | 93 | 4,335 |

Cross-split exact-content groups: train/validation 300, train/test 252,
validation/test 56. Four train/validation groups carry conflicting labels.

Near-duplicate analysis (TF-IDF cosine, threshold 0.90) finds **2,659 training
records (12.01%)** whose nearest neighbour is a near-copy.

**This is the most important caveat in the submission.** Server-level isolation
holds, but content-level leakage across splits does not, so validation macro F1
is probably optimistic for genuinely novel servers. It also puts a hard ceiling
on achievable accuracy: some identical tool texts carry different gold labels.

### 5.4 Target leakage

Category words do appear in names, descriptions, schemas and server slugs, but
that alone is not leakage — action verbs naturally occur in tool descriptions.

The real risk is `server_slug`: **580 of 1,448 training servers (40.06%)**
contain tools of a single category, so server identity is a powerful shortcut
that would not generalise to unseen servers. It is therefore **excluded from
model input**. The models see only `name`, `description` and `input_schema`.

### 5.5 Input representation

The baseline consumes the raw serialized schema, because word n-grams over raw
JSON (key names, type strings) are informative to a linear model:

```text
Tool name: <name>
Description: <description>
Input schema: <raw JSON schema>
```

The SLM gets the schema rendered as compact natural language — parameter names,
types, required/optional status, descriptions and enum values — which is far
more token-efficient and closer to its pretraining distribution:

```text
Tool name: <name>
Description: <description>
Schema description: <schema description>
Parameters:
- <param>: <type> (required) — <description> — allowed values: [...]
```

Token-length audit with the Qwen tokenizer over 1,000 sampled records: min 16,
mean 105.7, max 1,461 tokens, with only 3 of 1,000 exceeding 1,024. Max
sequence length is therefore **1,024 with right truncation**, which truncates
about 0.3% of inputs and always keeps the name and description — the most
discriminative fields — intact.

---

## 6. Baseline

TF-IDF (1–2 grams, `min_df=2`, `max_features=150000`, sublinear tf) plus
Logistic Regression (`liblinear`, `class_weight="balanced"`, `max_iter=1000`,
`random_state=42`). The vectoriser is fit on train only. Trains in **14.5 s**.

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Read | 0.9188 | 0.9279 | 0.9233 | 2817 |
| Write | 0.7974 | 0.8326 | 0.8146 | 1087 |
| Execute | 0.6601 | 0.5702 | 0.6119 | 235 |
| Destructive | 0.9500 | 0.8261 | 0.8837 | 253 |
| Financial | 0.5500 | 0.5000 | 0.5238 | 22 |
| Other | 0.5000 | 0.2000 | 0.2857 | 15 |

**Accuracy 0.8751 · Macro F1 0.6738 · Weighted F1 0.8737**

The baseline scores a non-zero `Other` F1 (0.2857) where the SLM scores 0.0 —
see §8.

---

## 7. SLM

### 7.1 Model and tokenizer

`Qwen/Qwen2.5-1.5B` with its native tokenizer, used as a **sequence
classifier** (six logits) rather than as constrained generation. Chosen because
it is open-weight, comfortably under the 3B cap, strong on code and schema-like
text, and fits a 16 GB T4 in FP16 with gradient checkpointing.

Classification rather than generation is what makes the invalid-output rate
**structurally zero**: a prediction is one `argmax` over six logits, so an
unparseable or out-of-vocabulary label is not representable. There is no
decoding, no sampling and no parsing step to fail. Greedy argmax is also
deterministic, so repeated runs give identical output.

### 7.2 LoRA configuration

`r=8`, `lora_alpha=16`, `lora_dropout=0.05`, targeting
`q_proj, k_proj, v_proj, o_proj`, with the randomly-initialised classification
head (`score`) trained and saved alongside.

**2,188,288 of 1,545,911,808 parameters trainable (0.1416%).**

### 7.3 Class imbalance

Class-weighted cross entropy with `weight(c) = N / (K · count(c))`:

| Category | Weight |
|---|---:|
| Read | 0.2620 |
| Write | 0.6793 |
| Execute | 3.1409 |
| Destructive | 2.9197 |
| Financial | 33.2477 |
| Other | 48.5592 |

Weighting alone was not enough to recover `Other` — see §8.

### 7.4 Training configuration

| Setting | Value |
|---|---|
| Epochs | 2 |
| Batch size | 2 per device × 8 accumulation = 16 effective |
| Learning rate | 2e-4, AdamW, weight decay 0.01, 100 warmup steps |
| Max sequence length | 1,024 (padded per batch, not to 1,024) |
| Precision | FP16 with gradient checkpointing |
| Seed / data seed | 42 / 42 |

### 7.5 Checkpoint selection and stopping rule

Evaluation runs once per epoch; the checkpoint with the highest **validation
macro F1** is restored via `load_best_model_at_end` and saved to
`outputs/slm/final_adapter/`. Recorded under `training` in `metrics.json`.

| Epoch | Step | Eval loss | Accuracy | Macro F1 |
|---:|---:|---:|---:|---:|
| 1 | 1384 | 0.3836 | 0.9198 | 0.7209 |
| 2 | 2768 | 0.4139 | 0.9352 | **0.7220** |

Selected: **checkpoint-2768 (epoch 2), macro F1 0.721979**.

Two honest caveats. The margin over epoch 1 is **0.0011 macro F1**, well within
noise for classes with 15–22 validation examples — the two checkpoints are
effectively tied on the primary metric, and epoch 2 was taken because it is
also clearly better on accuracy and weighted F1. And eval loss *rose* while
macro F1 rose, the usual signature of a class-weighted objective beginning to
overfit; a third epoch was not attempted.

### 7.6 FP16 training issue

The first run produced inf/NaN gradients: PEFT leaves the new trainable LoRA
and classification-head parameters in the base FP16 dtype, which the gradient
scaler cannot handle, and training stalled at the first step. Fixed by casting
trainable parameters back to FP32 after `get_peft_model`; the frozen base
weights stay FP16, which is where the memory saving actually comes from. The
fix is commented in `scripts/train_slm.py` so it is not accidentally reverted.

---

## 8. Results

Validation split, 4,429 records. Full numbers in `metrics.json`.

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Read | 0.9534 | 0.9656 | 0.9594 | 2817 |
| Write | 0.9034 | 0.9117 | 0.9075 | 1087 |
| Execute | 0.8426 | 0.7745 | 0.8071 | 235 |
| Destructive | 0.9630 | 0.9249 | 0.9435 | 253 |
| Financial | 0.7500 | 0.6818 | 0.7143 | 22 |
| Other | 0.0000 | 0.0000 | 0.0000 | 15 |

**Accuracy 0.9352 · Macro F1 0.7220 · Weighted F1 0.9332 · Invalid-output rate 0.0%**

### Confusion matrix

Table and image: `outputs/slm/confusion_matrix.csv` and `.png`
(baseline equivalents in `outputs/baseline/`).

| Actual \ Predicted | Read | Write | Execute | Destructive | Financial | Other |
|---|---:|---:|---:|---:|---:|---:|
| Read | 2720 | 70 | 19 | 6 | 2 | 0 |
| Write | 81 | 991 | 11 | 2 | 2 | 0 |
| Execute | 31 | 22 | 182 | 0 | 0 | 0 |
| Destructive | 6 | 9 | 3 | 234 | 1 | 0 |
| Financial | 1 | 4 | 1 | 1 | 15 | 0 |
| Other | 14 | 1 | 0 | 0 | 0 | 0 |

The `Other` column is entirely zero — the model never predicts the class.

### Cost

| | Baseline | SLM |
|---|---:|---:|
| Latency | 0.21 ms/example | 97.4 ms/example |
| Throughput | 4,745 /s | 10.3 /s |
| Artifact size | 13.0 MB | 8.4 MB adapter + 3.09 GB base |
| Peak GPU memory | n/a (CPU) | 6.37 GB |

The SLM buys +0.048 macro F1 for a **460× latency increase**. That tradeoff is
defensible for batch cataloguing, not for a hot path — see §10.

---

## 9. Error analysis

287 validation errors (6.48%). All of them in
`outputs/slm/validation_errors.csv`; failure modes and worked examples in
`outputs/slm/error_analysis.json`.

| Actual class | Errors |
|---|---:|
| Read | 97 |
| Write | 96 |
| Execute | 53 |
| Destructive | 19 |
| Other | 15 |
| Financial | 7 |

Examples below are the model's **highest-confidence** errors, which show what
it actually learned rather than where it was merely undecided.

### Failure mode 1 — `Read` vs `Write` (151 errors, 53% of all errors)

Write→Read 81, Read→Write 70. The model keys on surface wording rather than on
whether state changes.

- `browser_network_requests` — *"Returns all network requests since loading the
  page"* — gold `Write`, predicted `Read` at confidence 1.00. "Returns"
  dominates, though the tool drives a browser session.
- `seo.og_image.generate` — *"Generate an Open Graph image artifact and return
  hosted URL"* — gold `Read`, predicted `Write`. "Generate" reads as mutation.

These two have **opposite** gold labels for near-identical surface cues, which
is partly genuine label noise: §5.3 found 4 train/validation content groups
with conflicting labels.

**Fix:** derive a state-change feature from the schema (presence of `id` plus
payload fields, HTTP-verb-like parameter names) and train with hard negatives
pairing `Read`/`Write` tools from the same server.

### Failure mode 2 — `Execute` vs `Read`/`Write` (83 errors)

Execute→Read 31, Execute→Write 22, Read→Execute 19, Write→Execute 11.

- `browser_file_upload` — *"Upload a local file to a web page"* — gold
  `Execute`, predicted `Write`.
- `code_review_deep` — *"Review code for security, performance, and quality"*,
  no schema at all — gold `Execute`, predicted `Read`.

`Execute` is the vaguest category and several of these are arguably ambiguous
rather than wrong.

**Fix:** this is the strongest candidate for a multi-label formulation, since
"runs an operation *and* returns data" is genuinely two attributes being forced
into one label.

### Failure mode 3 — `Other` collapse (15 of 15 errors)

Every validation `Other` record is misclassified, 14 as `Read` and 1 as
`Write`. `glob`, `rpc_discover` and `lambda_function` are all predicted `Read`
at confidence ≥ 0.997.

Class weighting at 48.6× was not enough: with 76 training examples the model
never finds a coherent decision region, and `Other` is a semantic grab-bag with
no shared surface signal. The **linear baseline does better here** (F1 0.2857
vs 0.0), because TF-IDF can latch onto rare literal tokens that the weighted
fine-tune smooths away.

**Fix:** the honest one is that `Other` needs either an order of magnitude more
examples or an abstention route — predict `Other` when max softmax falls below
a calibrated threshold, rather than learning it as a positive class.

### Cross-cutting: confidence is not calibrated

Most errors above are emitted above 0.99 confidence. Raw softmax is not usable
as an abstention signal without temperature scaling on a held-out split.

---

## 10. Test predictions

`predictions.csv` — 4,428 rows, two columns, unique `record_id`s, one valid
category each, in input order. The test set was not used for training, tuning
or checkpoint selection.

Predicted distribution, stated plainly because it is asymmetric with training:

| Category | Count | Share | Train share |
|---|---:|---:|---:|
| Read | 2,897 | 65.4% | 63.6% |
| Write | 1,078 | 24.3% | 24.5% |
| Destructive | 253 | 5.7% | 5.7% |
| Execute | 188 | 4.2% | 5.3% |
| Financial | 12 | 0.3% | 0.5% |
| **Other** | **0** | **0.0%** | **0.3%** |

**Zero `Other` predictions is expected, not a bug** — it follows directly from
the validation collapse in §9. If the private test set has roughly the same
`Other` rate as train, about 15 records are guaranteed wrong, costing up to
~0.17 macro F1 on their own. `Execute` and `Financial` are also
under-predicted, consistent with their sub-1.0 recall.

---

## 11. Reproducibility

- All randomness seeded at 42 (`seed`, `data_seed`, `random_state`).
- Splits used exactly as supplied; no re-splitting.
- `scripts/train_slm.py` saves the selected adapter directly, so
  `outputs/slm/final_adapter/` is a script output, not a manual artifact.
- `predict.py` builds its input text through `scripts/data_prep.build_slm_text`,
  the same function training uses, so inference cannot drift from training. An
  earlier version of the inference code normalised null fields before building
  the text; because the adapter was trained on the un-normalised form, that
  changed 27 of 4,428 test predictions. The normalisation was removed and
  `predictions.csv` reflects the training-consistent text.
- Trainer checkpoints (optimizer, scheduler and RNG state) are excluded per the
  brief; only the ~8 MB LoRA adapter ships. `scripts/train_slm.py` regenerates
  them.

---

## 12. What I would do next

In rough order of expected macro F1 per unit of effort:

1. **Abstention for `Other`** — calibrate confidence and route low-confidence
   predictions there. Targets the single largest macro F1 loss without new data.
2. **Deduplicate against validation** and re-measure; 0.7220 is likely
   optimistic given the 12% near-duplicate rate.
3. **Hard-negative training** on `Read`/`Write`/`Execute` pairs from the same
   server, where 81% of errors live.
4. **Schema-derived state-change features** instead of relying on description
   wording.
5. **Distil to a smaller encoder** — the baseline already reaches 94% of the
   SLM's weighted F1 at 1/460th the latency, so a ~100M encoder likely captures
   most of the gain far more cheaply.
6. **Multi-label reformulation**, since `Execute`+`Read` is genuinely two
   attributes.

---

## 13. Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | README with setup, commands, decisions, results | this file |
| 2 | Reproducible data-preparation code | `scripts/data_prep.py`, `scripts/audit_data.py` |
| 3 | Baseline training/evaluation code | `scripts/train_baseline.py` |
| 4 | SLM training code | `scripts/train_slm.py` |
| 5 | Inference entry point | `predict.py` |
| 6 | Test predictions | `predictions.csv` |
| 7 | Metrics + confusion matrix | `metrics.json`, `outputs/slm/confusion_matrix.csv` and `.png` |
| 8 | Model card | `MODEL_CARD.md` |

Error analysis (required work §5) is in `outputs/slm/error_analysis.json` and
`validation_errors.csv`; the data audit (§1) is in
`outputs/audit/data_audit.json`.

Large base-model weights are not included. The model is reconstructed from the
public identifier `Qwen/Qwen2.5-1.5B` plus the supplied LoRA adapter.
