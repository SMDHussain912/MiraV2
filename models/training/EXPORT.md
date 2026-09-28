# TAMEV export recipe — PyTorch → ONNX → INT8 (Phase 7)

This is the reproducible path from a fine-tuned checkpoint to the artifact the
C++ runtime loads (`models/model_int8.onnx`). Nothing else in the repository
produces ONNX. All steps are **dev-only Python** (venv `~/ml_env` via `source
~/v.sh`); the C++ runtime never depends on that environment.

## 1. Build leakage-safe splits

```bash
source ~/v.sh
cd models/training
python3 build_splits.py
```

Reads `datasets/*.jsonl` + the curated `dataset.py` rows, dedups
(punct-insensitive, curated > handwritten > generated priority), groups
near-duplicates by template skeleton, balances to 90 rows/intent and writes
`train.jsonl` / `validation.jsonl` / `test.jsonl` / `corpus.jsonl` plus
`split_report.json` (hard-fails on exact-text or group-id overlap across splits).

## 2. Train

```bash
python3 train.py
```

Seeded (`SEED = 42`, per-epoch shuffle seed), fixed `padding="max_length"`
(ctx 128 / options 64 — same shapes as the runtime contract), saves the best
val-accuracy checkpoint to `mira-tamev/`.

## 3. Evaluate with per-intent metrics

```bash
python3 eval_model.py --model mira-tamev --data test.jsonl \
    --report reports/eval_test.json --show-errors 25
```

Prints per-intent precision/recall/F1, macro-F1, balanced accuracy, the
confusion matrix and misclassified utterances (shared label order from
`metrics.py`).

## 4. Export ONNX (fp32 → int8) and verify

```bash
python3 export_onnx.py
```

Exact parameters (hardcoded in `export_onnx.py`, echoed into
`reports/export_report.json`):

- exporter: `torch.onnx.export` (legacy tracer, `dynamo=False`), traced through
  a wrapper that returns `(logits, probs)` — the two outputs the C++ classifier
  consumes (it prefers `probs`, falls back to softmax of `logits`);
- opset **17**, `do_constant_folding=True`;
- inputs `ctx_input_ids`, `ctx_attention_mask` (dynamic `batch_size`, `ctx_len`),
  `opt_input_ids`, `opt_attention_mask` (dynamic `batch_size`, `num_options`,
  `opt_len`); outputs `logits`, `probs` (dynamic `batch_size`, `num_options`);
- int8: `onnxruntime.quantization.quantize_dynamic(weight_type=QInt8,
  per_channel=True, nodes_to_exclude=C*)` → `models/model_int8.onnx`.
  **Why the exclusions:** a plain full-graph quantisation collapses outputs
  toward `conversation`. The sensitivity sweep in
  `reports/quant_variants*.txt` (`quant_debug*.py`) shows layer.0 alone costs
  max|Δlogit| ≈ 9.5; excluding the 43 sensitive MatMuls of **C\*** — `layer.0/*`,
  all `attention/self/{query,key,value}` projections, and the head
  (`q/k_proj`, `pointer_head`) — keeps argmax agreement at 1.000 on every
  test row while still shrinking ~57 MB → ~31 MB (layers 1–3 attention-output
  and FFN MatMuls remain quantised).
  **Why QInt8:** QUInt8 per-channel is equivalent on PyTorch agreement
  (max|ΔP| 0.112 vs 0.108) but its cross-ORT-version skew — C++ ORT 1.29 vs
  fixture venv ORT 1.30 on the 7 parity vectors — is 2.2e-3, over the parity
  test's 1e-3 tolerance; QInt8 measures 2.6e-4 (QUInt8 reduceRange 3.8e-4,
  QU8+L3 3.6e-4, QI8+L3 7e-5 were the alternates measured);
- verification against the **full test split** (`--rows 0`), hard-fails otherwise:
  - fp32 ORT vs PyTorch: max |ΔP| ≤ **1e-3** and argmax agreement **1.0**;
  - int8 ORT vs PyTorch: max |ΔP| ≤ **1.5e-1**, argmax agreement **1.0**, and
    confidence-gate agreement **1.0** (top-1 ≥ 0.5 on both sides — the C++
    `from_probabilities()` Unknown gate must never flip). The 1.5e-1 budget is
    the measured quantisation noise (~1.1 logit units on a near-one-hot
    softmax); the *behavioural* gates (predicted class, Known/Unknown) are
    exact, which is what the runtime actually consumes;
  - dynamic-axes smoke test: batch=3, ctx_len=64, opt_len=48 must run and emit
    shape `(3, 7)`.

Outputs: `models/training/export/model_fp32.onnx`,
`models/model_int8.onnx`, `models/training/reports/export_report.json`
(recipe + versions + sha256 + verification numbers).

## 5. Re-run the Phase 6 parity harness against the new artifact

```bash
source ~/v.sh
python3 models/parity/regenerate.py     # rewrites frozen_vectors.json from models/model_int8.onnx
cmake --build build
ctest --test-dir build -R 'tamev_parity|tamev_classifier' --output-on-failure
```

`regenerate.py` records the generator ORT version; `tests/test_tamev_parity.cpp`
compares C++ (system ORT 1.29) against the fixtures (venv ORT 1.30): argmax
equality is the hard gate, per-entry probabilities within 1e-3.

Any future retrain: repeat steps 2 → 4 → 5 in that order.
