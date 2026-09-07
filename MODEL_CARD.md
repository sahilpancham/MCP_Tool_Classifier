# Model card: MCP tool action classifier

## Model details

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B` (open weights, 1.5B parameters) |
| Adaptation | LoRA (r=8, alpha=16, dropout=0.05) on `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Head | Sequence classification, six labels |
| Trainable parameters | 2,188,288 of 1,545,911,808 (0.1416%) |
| Shipped artifact | `outputs/slm/final_adapter/` (LoRA adapter + tokenizer, ~8 MB) |
| Training data | Only the supplied `data/train.jsonl` (22,143 records, 1,448 servers) |
| Primary metric | Macro F1 |

The base weights are not included. Loading the adapter downloads
`Qwen/Qwen2.5-1.5B` from the Hugging Face Hub.

## Intended use

Classify a single MCP (Model Context Protocol) tool definition into exactly one
of `Read`, `Write`, `Execute`, `Destructive`, `Financial`, `Other`, using the
tool's `name`, `description` and `input_schema`.

Intended as a triage and cataloguing aid for MCP tool registries: labelling
large numbers of tools, surfacing likely high-risk tools for human review, and
routing tools into policy tiers.

## Out-of-scope use

**This model must not be used on its own as an authorization or access-control
decision.** A tool predicted `Read` may still mutate or delete state. Any
security-relevant use requires deterministic policy rules, an allowlist, or
human review in front of the model's output.

Also out of scope: multi-label tools (the model emits exactly one label),
non-MCP API definitions, and tools whose behaviour is not described in their
text or schema.

## Performance

Measured on the supplied validation split (4,429 records, 292 servers disjoint
from train). Full numbers in `metrics.json`.

| Metric | Baseline (TF-IDF + LR) | SLM |
|---|---:|---:|
| Macro F1 | 0.6738 | 0.7220 |
| Accuracy | 0.8751 | 0.9352 |
| Weighted F1 | 0.8737 | 0.9332 |

Per-class F1 for the SLM: Read 0.9594, Destructive 0.9435, Write 0.9075,
Execute 0.8071, Financial 0.7143, Other 0.0000.

Inference costs roughly 97 ms per example (batch size 8, max length 1,024) at
about 6.4 GB peak GPU memory, versus 0.21 ms per example for the linear
baseline.

## Known failure modes

1. **`Other` is never predicted.** With 76 training examples (0.34%), the class
   collapses entirely: all 15 validation `Other` records were misclassified,
   14 of them as `Read`. The `Other` F1 of 0.0 is the single largest drag on
   macro F1, and no `Other` label appears in the test predictions either.
2. **`Read` versus `Write` confusion (151 validation errors).** Tools that
   return data after performing an action, and tools described as "returns
   ..." while actually mutating state, are frequently swapped.
3. **`Execute` versus `Read`/`Write` confusion (83 validation errors).**
   Browser actions, workflows and analysis tools are labelled by surface
   wording rather than by whether they run an operation.
4. **`Financial` recall is 0.68 on 22 validation examples.** Refunds and
   deposits are often absorbed into `Write`. The class has only 111 training
   examples, so this estimate is high-variance.
5. **Missing inputs degrade quietly.** 16% of records have no `input_schema`
   and the model still emits a confident label from the name and description
   alone.
6. **Confidence is not calibrated.** Most errors are emitted at softmax
   confidence above 0.99, so the raw score is not usable as an abstention
   signal without post-hoc calibration.

## Data and evaluation limitations

- Splits are isolated by `server_slug`, but content-level duplication remains:
  300 exact-content groups are shared between train and validation and 252
  between train and test, and 12% of training records have a near-duplicate
  (cosine similarity >= 0.90). Validation macro F1 is therefore likely
  optimistic relative to genuinely novel servers.
- 40% of training servers contain only one category, so `server_slug` is a
  strong shortcut. It is deliberately excluded from model input.
- Some duplicated contents carry conflicting labels, which puts a ceiling on
  achievable accuracy.
- The held-out test set has no labels and was not used for any tuning or
  checkpoint selection.

## Ethical and operational considerations

Misclassifying a `Destructive` or `Financial` tool as `Read` is far more costly
than the reverse. Deployments should bias toward the high-risk class, pair the
model with deterministic rules for known dangerous verbs, monitor the predicted
label distribution for drift as new MCP servers appear, and route low-confidence
or high-risk predictions to human review.
