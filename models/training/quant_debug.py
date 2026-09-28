#!/usr/bin/env python3
"""Scratch: compare int8 quantisation variants against the fp32 export (Phase 7 debug)."""
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
pa = a.run(["probs"], inp)[0]

variants = [
    ("QUInt8 pc=True", dict(weight_type=QuantType.QUInt8, per_channel=True)),
    ("QUInt8 pc=False", dict(weight_type=QuantType.QUInt8, per_channel=False)),
    ("QInt8 pc=True MatMul", dict(weight_type=QuantType.QInt8, per_channel=True,
                                  op_types_to_quantize=["MatMul"])),
    ("QInt8 pc=False MatMul", dict(weight_type=QuantType.QInt8, per_channel=False,
                                   op_types_to_quantize=["MatMul"])),
    ("QUInt8 pc=True MatMul", dict(weight_type=QuantType.QUInt8, per_channel=True,
                                   op_types_to_quantize=["MatMul"])),
]
lines = []
for i, (name, kw) in enumerate(variants):
    out = f"/tmp/qv{i}.onnx"
    try:
        quantize_dynamic(model_input=FP, model_output=out, **kw)
        s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        pb = s.run(["probs"], inp)[0]
        d = float(np.abs(pa - pb).max())
        ag = float((pa.argmax(1) == pb.argmax(1)).mean())
        lines.append(f"{name:24} maxdiff={d:.4f} agree={ag:.3f}")
    except Exception as exc:
        lines.append(f"{name:24} ERROR {type(exc).__name__}: {str(exc)[:150]}")

with open("reports/quant_variants.txt", "w") as handle:
    handle.write("\n".join(lines) + "\n")
print("\n".join(lines))
