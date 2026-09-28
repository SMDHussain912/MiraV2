#!/usr/bin/env python3
"""Scratch: final quantisation tuning on combo C* (Phase 7 debug)."""
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
rows = [json.loads(l)["text"] for l in open("test.jsonl")]
ctx = [Q + "\nUser request: " + t for t in rows]
c = tok(ctx, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
o = tok(OPTS, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
oidx = o["input_ids"].unsqueeze(0).expand(len(rows), -1, -1)
omask = o["attention_mask"].unsqueeze(0).expand(len(rows), -1, -1)
inp = {"ctx_input_ids": c["input_ids"].numpy(),
       "ctx_attention_mask": c["attention_mask"].numpy(),
       "opt_input_ids": oidx.numpy(), "opt_attention_mask": omask.numpy()}
a = ort.InferenceSession(FP, providers=["CPUExecutionProvider"])
pa, la = a.run(["probs", "logits"], inp)

model = onnx.load(FP)
matmuls = [n.name for n in model.graph.node if n.op_type == "MatMul"]

def sel(*s):
    return [n for n in matmuls if all(x in n for x in s)]

def sela(*s):
    return [n for n in matmuls if any(x in n for x in s)]

head = sela("q_proj", "k_proj", "pointer_head")
layer0 = sel("layer.0/")
allqk = sel("attention/self/query") + sel("attention/self/key")
allqkv = allqk + sel("attention/self/value")
CSTAR = layer0 + allqkv + head
l1ffn = sel("layer.1/", "intermediate") + sel("layer.1/", "output")

variants = [
    ("C* QI8 pc=True", CSTAR, dict(weight_type=QuantType.QInt8, per_channel=True)),
    ("C* QI8 pc=False", CSTAR, dict(weight_type=QuantType.QInt8, per_channel=False)),
    ("C* QU8 reduceRange", CSTAR, dict(weight_type=QuantType.QUInt8, per_channel=True,
                                        reduce_range=True)),
    ("C*+L1ffn QU8", CSTAR + l1ffn, dict(weight_type=QuantType.QUInt8, per_channel=True)),
]
lines = []
for name, ex, kw in variants:
    out = "/tmp/qf_" + name.replace(" ", "_").replace("*", "star") + ".onnx"
    try:
        quantize_dynamic(model_input=FP, model_output=out,
                         nodes_to_exclude=ex, **kw)
        s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        pb, lb = s.run(["probs", "logits"], inp)
        d = float(np.abs(pa - pb).max())
        dl = float(np.abs(la - lb).max())
        agree = float((pa.argmax(1) == pb.argmax(1)).mean())
        # threshold-0.5 decision agreement (C++ Unknown gate)
        da = float(((pa.max(1) >= 0.5) == (pb.max(1) >= 0.5)).mean())
        minconf = float(pb.max(1).min())
        lines.append(f"{name:22} maxdP={d:.4f} maxdLogit={dl:.3f} agree={agree:.3f} "
                     f"thr50={da:.3f} minInt8conf={minconf:.3f} size={os.path.getsize(out)/1e6:.1f}MB")
    except Exception as exc:
        lines.append(f"{name:22} ERROR {type(exc).__name__}: {str(exc)[:120]}")

os.makedirs("reports", exist_ok=True)
open("reports/quant_variants7.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
