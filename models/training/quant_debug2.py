#!/usr/bin/env python3
"""Scratch: quantisation variants with nodes_to_exclude (Phase 7 debug)."""
import json
import logging

import numpy as np
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

# node names to exclude
import onnx
model = onnx.load(FP)
all_matmuls = [n.name for n in model.graph.node if n.op_type == "MatMul"]
head = [n for n in all_matmuls if "pointer_head" in n]
proj = [n for n in all_matmuls if "q_proj" in n or "k_proj" in n]
attn_qk = [n for n in all_matmuls if "attention/self/query" in n or "attention/self/key" in n]
print("head:", head)
print("proj:", proj)

QU8 = dict(weight_type=QuantType.QUInt8, per_channel=True)
variants = [
    ("QU8 head+proj excluded", dict(**QU8, nodes_to_exclude=head + proj)),
    ("QU8 head only excluded", dict(**QU8, nodes_to_exclude=head)),
    ("QU8 proj only excluded", dict(**QU8, nodes_to_exclude=proj)),
    ("QU8 head+proj+attnQK ex", dict(**QU8, nodes_to_exclude=head + proj + attn_qk)),
    ("QI8 pc=False head+proj ex", dict(weight_type=QuantType.QInt8, per_channel=False,
                                       nodes_to_exclude=head + proj)),
    ("QU8 reduceRange head+proj", dict(**QU8, reduce_range=True,
                                       nodes_to_exclude=head + proj)),
]
lines = []
for i, (name, kw) in enumerate(variants):
    out = f"/tmp/qx{i}.onnx"
    try:
        quantize_dynamic(model_input=FP, model_output=out, **kw)
        s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        pb, lb = s.run(["probs", "logits"], inp)
        d = float(np.abs(pa - pb).max())
        dl = float(np.abs(la - lb).max())
        agree = float((pa.argmax(1) == pb.argmax(1)).mean())
        conf_ok = float((pb.max(1) >= pa.max(1) - 0.1).mean())
        lines.append(f"{name:28} maxdP={d:.4f} maxdLogit={dl:.3f} "
                     f"agree={agree:.3f} confPreserved={conf_ok:.3f}")
    except Exception as exc:
        lines.append(f"{name:28} ERROR {type(exc).__name__}: {str(exc)[:150]}")

with open("reports/quant_variants2.txt", "w") as handle:
    handle.write("\n".join(lines) + "\n")
print("\n".join(lines))
