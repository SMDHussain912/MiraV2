#!/usr/bin/env python3
"""Scratch: round 5 exclusion combos (Phase 7 debug)."""
import json
import logging
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

logging.disable(logging.WARNING)

Q = "Which action should Mira perform?"
OPTS = ["open_application", "search_web", "read_screen", "type_text",
        "file_operation", "system_control", "conversation"]
FP = "export/model_fp32.onnx"

tok = AutoTokenizer.from_pretrained("mira-tamev", local_files_only=True,
                                    trust_remote_code=True)
rows = [json.loads(l)["text"] for l in open("test.jsonl")][:48]
ctx = [Q + "\nUser request: " + t for t in rows]
c = tok(ctx, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
o = tok(OPTS, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
oidx = o["input_ids"].unsqueeze(0).expand(len(rows), -1, -1)
omask = o["attention_mask"].unsqueeze(0).expand(len(rows), -1, -1)
inp = {
    "ctx_input_ids": c["input_ids"].numpy(),
    "ctx_attention_mask": c["attention_mask"].numpy(),
    "opt_input_ids": oidx.numpy(),
    "opt_attention_mask": omask.numpy(),
}
a = ort.InferenceSession(FP, providers=["CPUExecutionProvider"])
pa, la = a.run(["probs", "logits"], inp)

model = onnx.load(FP)
matmuls = [n.name for n in model.graph.node if n.op_type == "MatMul"]


def sel(*subs):
    return [n for n in matmuls if all(s in n for s in subs)]


def sel_any(*subs):
    return [n for n in matmuls if any(s in n for s in subs)]


head = sel_any("q_proj", "k_proj", "pointer_head")
layer0 = sel("layer.0/")
all_qk = sel("attention/self/query") + sel("attention/self/key")
all_qkv = all_qk + sel("attention/self/value")
all_attn_dense = sel("attention/output/dense")
ffn_rest = sel("layer.1/", "intermediate") + sel("layer.1/", "output") + \
    sel("layer.2/", "intermediate") + sel("layer.2/", "output") + \
    sel("layer.3/", "intermediate") + sel("layer.3/", "output")

QU8 = dict(weight_type=QuantType.QUInt8, per_channel=True)
variants = [
    ("L0+QK+head", layer0 + all_qk + head),
    ("L0+QKV+head", layer0 + all_qkv + head),
    ("L0+attnD+QK+head", layer0 + all_attn_dense + all_qk + head),
    ("L0+QKV+attnD+head", layer0 + all_qkv + all_attn_dense + head),
    ("onlyFFN L1-3 quant", layer0 + all_qkv + all_attn_dense + head + []),
    ("L0+QK+head QI8pcF", layer0 + all_qk + head),
]
lines = []
for i, (name, ex) in enumerate(variants):
    out = f"/tmp/qd{i}.onnx"
    kw = dict(weight_type=QuantType.QUInt8, per_channel=True)
    if "QI8pcF" in name:
        kw = dict(weight_type=QuantType.QInt8, per_channel=False)
    try:
        quantize_dynamic(model_input=FP, model_output=out,
                         nodes_to_exclude=ex, **kw)
        s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        pb, lb = s.run(["probs", "logits"], inp)
        d = float(np.abs(pa - pb).max())
        dl = float(np.abs(la - lb).max())
        agree = float((pa.argmax(1) == pb.argmax(1)).mean())
        mb = os.path.getsize(out) / 1e6
        lines.append(f"{name:20} excl={len(ex):2d} maxdP={d:.4f} maxdLogit={dl:.3f} "
                     f"agree={agree:.3f} size={mb:.1f}MB")
    except Exception as exc:
        lines.append(f"{name:20} ERROR {type(exc).__name__}: {str(exc)[:120]}")

with open("reports/quant_variants5.txt", "w") as handle:
    handle.write("\n".join(lines) + "\n")
print("\n".join(lines))
