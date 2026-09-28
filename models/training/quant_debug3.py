#!/usr/bin/env python3
"""Scratch: per-layer quantisation sensitivity (Phase 7 debug)."""
import json
import logging

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
print("fp32 logits: min %.3f max %.3f range %.3f" % (la.min(), la.max(), float(np.ptp(la))))

model = onnx.load(FP)
matmuls = [n.name for n in model.graph.node if n.op_type == "MatMul"]
# what got quantized in the baseline?
base = "/tmp/qlayer_base.onnx"
quantize_dynamic(model_input=FP, model_output=base,
                 weight_type=QuantType.QUInt8, per_channel=True)
qm = onnx.load(base)
qnames = {n.name for n in qm.graph.node if n.op_type == "MatMulInteger"}
print("quantized baseline nodes (%d):" % len(qnames))
for n in sorted(qnames):
    print("   ", n)

QU8 = dict(weight_type=QuantType.QUInt8, per_channel=True)
lines = []
for layer in range(4):
    ex = [n for n in matmuls if f"layer.{layer}/" in n]
    out = f"/tmp/qlayer{layer}.onnx"
    try:
        quantize_dynamic(model_input=FP, model_output=out,
                         nodes_to_exclude=ex, **QU8)
        s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        pb, lb = s.run(["probs", "logits"], inp)
        d = float(np.abs(pa - pb).max())
        dl = float(np.abs(la - lb).max())
        agree = float((pa.argmax(1) == pb.argmax(1)).mean())
        lines.append(f"exclude layer.{layer} ({len(ex)} nodes)  maxdP={d:.4f} "
                     f"maxdLogit={dl:.3f} agree={agree:.3f}")
    except Exception as exc:
        lines.append(f"exclude layer.{layer} ERROR {type(exc).__name__}: {str(exc)[:120]}")

with open("reports/quant_variants3.txt", "w") as handle:
    handle.write("\n".join(lines) + "\n")
print("\n".join(lines))
