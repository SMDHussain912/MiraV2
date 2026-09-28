#!/usr/bin/env python3
"""Reproducible PyTorch -> ONNX (fp32 -> int8) export for the fine-tuned TAMEV (Phase 7).

Offline dev step only — never runs in the C++ runtime.

    source ~/ml_env/bin/activate
    cd models/training
    python3 export_onnx.py

Produces:
    export/model_fp32.onnx          traced fp32 graph (dynamic batch/ctx_len/opt_len)
    ../model_int8.onnx              dynamic-quantised artifact the C++ runtime loads
    reports/export_report.json      recipe + verification numbers

Verification: the exported graphs are run with onnxruntime against PyTorch outputs on
the full held-out test split (max-abs-diff + argmax agreement + 0.5 confidence-gate
agreement) and with ragged shapes to prove the dynamic axes. Sensitive MatMul nodes
(layer.0, query/key/value projections, pointer head) are kept in fp32 — sensitivity
sweep in reports/quant_variants*.txt. Recipe documentation: models/training/EXPORT.md.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import metrics  # noqa: E402  (shared label order)

QUESTION = "Which action should Mira perform?"
OPTIONS = metrics.INTENTS
CTX_LEN = 128
OPT_LEN = 64
OPSET = 17

# Verification budgets (Phase 7 completion: exported file agrees with PyTorch).
FP32_MAX_ABS_TOL = 1e-3      # fp32 ORT vs PyTorch probability vectors
# int8 dynamic quantisation leaves ~1.1 logit-unit noise on this model (measured;
# sensitivity sweep: reports/quant_variants4-7.txt). On the near-one-hot softmax
# outputs that costs <= ~0.10 in probability while the predicted class and the
# C++ 0.5 confidence gate never change — so the class/gate gates are exact.
INT8_MAX_ABS_TOL = 1.5e-1    # int8 ORT vs PyTorch (quantisation noise budget)
INT8_MIN_ARGMAX_AGREEMENT = 1.0      # class prediction must never flip
INT8_MIN_GATE_AGREEMENT = 1.0        # confidence>=0.5 (C++ Unknown gate) must never flip


class ExportWrapper(torch.nn.Module):
    """Expose only the two graph outputs the C++ classifier can consume."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ctx_input_ids, ctx_attention_mask, opt_input_ids, opt_attention_mask):
        out = self.model(
            ctx_input_ids=ctx_input_ids,
            ctx_attention_mask=ctx_attention_mask,
            opt_input_ids=opt_input_ids,
            opt_attention_mask=opt_attention_mask,
        )
        return out["logits"], out["probs"]


def encode(tokenizer, texts, max_length):
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def build_batch(tokenizer, rows):
    """Rows (list of text) -> the four canonical int64 input tensors."""
    contexts = [QUESTION + "\nUser request: " + text for text in rows]
    ctx_ids, ctx_mask = encode(tokenizer, contexts, CTX_LEN)
    opt_ids, opt_mask = encode(tokenizer, OPTIONS, OPT_LEN)
    opt_ids = opt_ids.unsqueeze(0).expand(len(rows), -1, -1).contiguous()
    opt_mask = opt_mask.unsqueeze(0).expand(len(rows), -1, -1).contiguous()
    return ctx_ids, ctx_mask, opt_ids, opt_mask


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantisation_exclusions(model):
    """MatMul nodes kept in fp32 during int8 dynamic quantisation.

    Sensitivity sweep (quant_debug3-5, reports/quant_variants{3,4,5,7}.txt):
    quantising layer.0 alone costs max|dLogit| ~9.5; excluding layer.0 drops it
    to ~1.4, and layer.0 + query/key/value projections + pointer head gives
    exact argmax agreement with fp32 on every test row while the graph still
    shrinks ~57 MB -> ~31 MB. Remaining quantised nodes are the layers 1-3
    attention-output and feed-forward MatMuls of the context encoder.

    QInt8 per-channel over QUInt8: identical behavioural agreement, but its
    cross-ORT-version skew (C++ ORT 1.29 vs fixture venv ORT 1.30 on the 7
    parity vectors) is 2.6e-4 vs 2.2e-3 for QUInt8 — the Phase 6 parity test
    requires < 1e-3.
    """
    matmuls = [n.name for n in model.graph.node if n.op_type == "MatMul"]
    layer0 = [n for n in matmuls if "layer.0/" in n]
    qkv = [n for n in matmuls
           if any(tag in n for tag in ("attention/self/query",
                                       "attention/self/key",
                                       "attention/self/value"))]
    head = [n for n in matmuls
            if any(tag in n for tag in ("q_proj", "k_proj", "pointer_head"))]
    return layer0 + qkv + head


def pytorch_probs(model, batch):
    model.eval()
    with torch.no_grad():
        out = model(
            ctx_input_ids=batch[0],
            ctx_attention_mask=batch[1],
            opt_input_ids=batch[2],
            opt_attention_mask=batch[3],
        )
    return out["probs"].cpu().numpy(), out["logits"].cpu().numpy()


def ort_probs(session, batch):
    inputs = {
        "ctx_input_ids": batch[0].numpy().astype(np.int64),
        "ctx_attention_mask": batch[1].numpy().astype(np.int64),
        "opt_input_ids": batch[2].numpy().astype(np.int64),
        "opt_attention_mask": batch[3].numpy().astype(np.int64),
    }
    logits, probs = session.run(["logits", "probs"], inputs)
    return probs, logits


def compare(reference, candidate):
    """Probs arrays -> {max_abs_diff, mean_abs_diff, argmax_agreement}."""
    ref_arg = reference.argmax(axis=1)
    cand_arg = candidate.argmax(axis=1)
    return {
        "max_abs_diff": float(np.abs(reference - candidate).max()),
        "mean_abs_diff": float(np.abs(reference - candidate).mean()),
        "argmax_agreement": float((ref_arg == cand_arg).mean()),
        "n": int(reference.shape[0]),
    }
def load_rows(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line)["text"])
            if limit and len(rows) >= limit:
                break
    return rows


def export_torch(model, fp32_path):
    wrapper = ExportWrapper(model).eval()
    example = (
        torch.ones(1, CTX_LEN, dtype=torch.long),
        torch.ones(1, CTX_LEN, dtype=torch.long),
        torch.ones(1, len(OPTIONS), OPT_LEN, dtype=torch.long),
        torch.ones(1, len(OPTIONS), OPT_LEN, dtype=torch.long),
    )
    dynamic_axes = {
        "ctx_input_ids": {0: "batch_size", 1: "ctx_len"},
        "ctx_attention_mask": {0: "batch_size", 1: "ctx_len"},
        "opt_input_ids": {0: "batch_size", 1: "num_options", 2: "opt_len"},
        "opt_attention_mask": {0: "batch_size", 1: "num_options", 2: "opt_len"},
        "logits": {0: "batch_size", 1: "num_options"},
        "probs": {0: "batch_size", 1: "num_options"},
    }
    kwargs = dict(
        input_names=["ctx_input_ids", "ctx_attention_mask",
                     "opt_input_ids", "opt_attention_mask"],
        output_names=["logits", "probs"],
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(wrapper, example, fp32_path, dynamo=False, **kwargs)
        exporter = "torch.onnx.export (legacy tracer, dynamo=False)"
    except TypeError:
        # older torch without the dynamo kwarg
        torch.onnx.export(wrapper, example, fp32_path, **kwargs)
        exporter = "torch.onnx.export (legacy tracer)"
    return exporter, list(dynamic_axes.keys())


def write_report(path, report):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
        handle.write("\n")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.path.join(HERE, "mira-tamev"))
    parser.add_argument("--data", default=os.path.join(HERE, "test.jsonl"),
                        help="JSONL rows used for the verification batch")
    parser.add_argument("--rows", type=int, default=0,
                        help="cap on verification rows (0 = all rows of the split)")
    parser.add_argument("--fp32-out", default=os.path.join(HERE, "export", "model_fp32.onnx"))
    parser.add_argument("--int8-out", default=os.path.join(REPO, "models", "model_int8.onnx"))
    parser.add_argument("--report", default=os.path.join(HERE, "reports", "export_report.json"))
    parser.add_argument("--skip-quant", action="store_true",
                        help="stop after the fp32 export (debugging)")
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"checkpoint: {args.checkpoint}")
    print(f"torch {torch.__version__} | onnx {onnx.__version__} | onnxruntime {ort.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(args.checkpoint, local_files_only=True,
                                      trust_remote_code=True).eval()

    rows = load_rows(args.data, args.rows)
    batch = build_batch(tokenizer, rows)
    ref_probs, ref_logits = pytorch_probs(model, batch)
    print(f"verification batch: {len(rows)} rows from {os.path.basename(args.data)}")

    # ------------------------------------------------------------------
    # 1. fp32 export
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.fp32_out)), exist_ok=True)
    exporter, dynamic_names = export_torch(model, args.fp32_out)
    print(f"exported fp32 -> {args.fp32_out}")

    fp32_session = ort.InferenceSession(args.fp32_out, providers=["CPUExecutionProvider"])
    fp32_probs, fp32_logits = ort_probs(fp32_session, batch)
    fp32_check = compare(ref_probs, fp32_probs)
    fp32_logit_check = compare(ref_logits, fp32_logits)
    print(f"fp32 ORT vs PyTorch : max|dP|={fp32_check['max_abs_diff']:.2e} "
          f"argmax={fp32_check['argmax_agreement']:.3f}")
    if fp32_check["max_abs_diff"] > FP32_MAX_ABS_TOL or fp32_check["argmax_agreement"] < 1.0:
        raise SystemExit(
            f"FAIL: fp32 export disagrees with PyTorch: {fp32_check} (tol {FP32_MAX_ABS_TOL})"
        )
    # Sensitive MatMuls to keep fp32 in the int8 artifact (see quantisation_exclusions).
    excluded = quantisation_exclusions(onnx.load(args.fp32_out))
    report = {
        "recipe": {
            "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), REPO),
            "command": "python3 export_onnx.py",
            "exporter": exporter,
            "opset": OPSET,
            "do_constant_folding": True,
            "dynamic_axes": dynamic_names,
            "inputs": ["ctx_input_ids", "ctx_attention_mask",
                       "opt_input_ids", "opt_attention_mask"],
            "outputs": ["logits", "probs"],
            "ctx_len": CTX_LEN,
            "opt_len": OPT_LEN,
            "options": OPTIONS,
            "quantization": None if args.skip_quant else (
                "onnxruntime.quantization.quantize_dynamic("
                "weight_type=QInt8, per_channel=True, nodes_to_exclude=C* "
                "[layer.0, attention/self/{query,key,value}, q/k_proj, pointer_head])"
            ),
            "quantization_exclusions": [] if args.skip_quant else excluded,
        },
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "files": {
            "fp32": {
                "path": os.path.relpath(os.path.abspath(args.fp32_out), REPO),
                "bytes": os.path.getsize(args.fp32_out),
                "sha256": sha256(args.fp32_out),
            },
        },
        "verification": {
            "batch_rows": len(rows),
            "fp32_ort_vs_pytorch_probs": fp32_check,
            "fp32_ort_vs_pytorch_logits": fp32_logit_check,
        },
        "budgets": {
            "fp32_max_abs_tol": FP32_MAX_ABS_TOL,
            "int8_max_abs_tol": INT8_MAX_ABS_TOL,
            "int8_min_argmax_agreement": INT8_MIN_ARGMAX_AGREEMENT,
            "int8_min_gate_agreement": INT8_MIN_GATE_AGREEMENT,
        },
    }

    if args.skip_quant:
        print("skipping quantisation (--skip-quant)")
        write_report(args.report, report)
        return 0

    # ------------------------------------------------------------------
    # 2. int8 dynamic quantisation -> artifact the C++ runtime loads
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.int8_out)), exist_ok=True)
    quantize_dynamic(
        model_input=args.fp32_out,
        model_output=args.int8_out,
        nodes_to_exclude=excluded,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"quantised int8 ({len(excluded)} sensitive MatMuls kept fp32) "
          f"-> {args.int8_out}")

    # ------------------------------------------------------------------
    # 3. verify int8 artifact + dynamic-axes smoke test
    # ------------------------------------------------------------------
    int8_session = ort.InferenceSession(args.int8_out, providers=["CPUExecutionProvider"])
    int8_probs, int8_logits = ort_probs(int8_session, batch)
    int8_check = compare(ref_probs, int8_probs)
    int8_logit_check = compare(ref_logits, int8_logits)
    # C++ from_probabilities() returns Unknown below confidence 0.5 — the gate
    # must agree between PyTorch and int8 or runtime behaviour would change.
    gate_ref = ref_probs.max(axis=1) >= 0.5
    gate_int8 = int8_probs.max(axis=1) >= 0.5
    gate_agreement = float((gate_ref == gate_int8).mean())
    min_int8_conf = float(int8_probs.max(axis=1).min())
    print(f"int8 ORT vs PyTorch : max|dP|={int8_check['max_abs_diff']:.4f} "
          f"argmax={int8_check['argmax_agreement']:.3f} "
          f"gate={gate_agreement:.3f} minInt8conf={min_int8_conf:.3f}")

    # Dynamic axes: ragged batch/sequence shapes must load and run.
    ragged_rows = rows[:3]
    contexts = [QUESTION + "\nUser request: " + text for text in ragged_rows]
    ctx_ids, ctx_mask = encode(tokenizer, contexts, 64)
    opt_ids, opt_mask = encode(tokenizer, OPTIONS, 48)
    ragged = (ctx_ids, ctx_mask,
              opt_ids.unsqueeze(0).expand(len(ragged_rows), -1, -1).contiguous(),
              opt_mask.unsqueeze(0).expand(len(ragged_rows), -1, -1).contiguous())
    ragged_probs, _ = ort_probs(int8_session, ragged)
    ragged_ok = ragged_probs.shape == (len(ragged_rows), len(OPTIONS))
    print(f"dynamic-axes smoke  : batch=3 ctx_len=64 opt_len=48 -> {tuple(ragged_probs.shape)} "
          f"{'OK' if ragged_ok else 'FAIL'}")
    if not ragged_ok:
        raise SystemExit("FAIL: dynamic axes do not produce the expected output shape")

    report["files"]["int8"] = {
        "path": os.path.relpath(os.path.abspath(args.int8_out), REPO),
        "bytes": os.path.getsize(args.int8_out),
        "sha256": sha256(args.int8_out),
    }
    report["verification"]["int8_ort_vs_pytorch_probs"] = int8_check
    report["verification"]["int8_ort_vs_pytorch_logits"] = int8_logit_check
    report["verification"]["int8_confidence_gate_agreement"] = gate_agreement
    report["verification"]["int8_min_top1_confidence"] = min_int8_conf
    report["verification"]["dynamic_axes_smoke"] = {
        "batch": len(ragged_rows),
        "ctx_len": 64,
        "opt_len": 48,
        "output_shape": list(ragged_probs.shape),
        "ok": bool(ragged_ok),
    }

    failures = []
    if int8_check["max_abs_diff"] > INT8_MAX_ABS_TOL:
        failures.append(f"int8 max abs diff {int8_check['max_abs_diff']:.4f} > {INT8_MAX_ABS_TOL}")
    if int8_check["argmax_agreement"] < INT8_MIN_ARGMAX_AGREEMENT:
        failures.append(f"int8 argmax agreement {int8_check['argmax_agreement']:.3f} "
                        f"< {INT8_MIN_ARGMAX_AGREEMENT}")
    if gate_agreement < INT8_MIN_GATE_AGREEMENT:
        failures.append(f"int8 confidence-gate agreement {gate_agreement:.3f} "
                        f"< {INT8_MIN_GATE_AGREEMENT}")
    if failures:
        write_report(args.report, report)
        raise SystemExit("FAIL: " + "; ".join(failures))
    print("verification: PASS")

    write_report(args.report, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
