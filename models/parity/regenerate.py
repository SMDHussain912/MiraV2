#!/usr/bin/env python3
"""Regenerate models/parity/frozen_vectors.json (Phase 6 parity fixture).

Usage (from the miraV2 repo root):
    source ~/v.sh   # activates ~/ml_env (needs onnxruntime, tokenizers, numpy)
    python3 models/parity/regenerate.py

Uses the canonical Phase 6 inputs:
  context = "Which action should Mira perform?\\nUser request: " + utterance
  options = 7-entry catalog in tamev_option_strings() training order.
Writes the generator onnxruntime version into the JSON so ORT-version skew
(venv vs system lib the C++ binary links) stays visible.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # models/
REPO = os.path.dirname(BASE)
PARITY_DIR = os.path.join(BASE, "parity")
OUT_PATH = os.path.join(PARITY_DIR, "frozen_vectors.json")

QUESTION = "Which action should Mira perform?"
OPTIONS = [
    "open_application",
    "search_web",
    "read_screen",
    "type_text",
    "file_operation",
    "system_control",
    "conversation",
]
CTX_LEN = 128
OPT_LEN = 64

UTTERANCES = [
    "Open Firefox",
    "Search for Python tutorials",
    "What does this window say?",
    "Type hello world",
    "Move my essay to Documents",
    "Turn the volume down",
    "Tell me a joke",
]


def encode(tokenizer: Tokenizer, text: str, max_len: int):
    enc = tokenizer.encode(text)
    ids = enc.ids[:max_len]
    mask = [1] * len(ids)
    pad = max_len - len(ids)
    return ids + [0] * pad, mask + [0] * pad


def main() -> int:
    print(f"onnxruntime: {ort.__version__}")
    tokenizer = Tokenizer.from_file(
        os.path.join(REPO, "models", "tamev-base", "tokenizer.json")
    )
    session = ort.InferenceSession(
        os.path.join(REPO, "models", "model_int8.onnx"),
        providers=["CPUExecutionProvider"],
    )

    vectors = []
    for utterance in UTTERANCES:
        context = QUESTION + "\nUser request: " + utterance
        ctx_ids, ctx_mask = encode(tokenizer, context, CTX_LEN)
        opt_ids, opt_masks = [], []
        for option in OPTIONS:
            ids, mask = encode(tokenizer, option, OPT_LEN)
            opt_ids.append(ids)
            opt_masks.append(mask)

        inputs = {
            "ctx_input_ids": np.array([ctx_ids], dtype=np.int64),
            "ctx_attention_mask": np.array([ctx_mask], dtype=np.int64),
            "opt_input_ids": np.array([opt_ids], dtype=np.int64),
            "opt_attention_mask": np.array([opt_masks], dtype=np.int64),
        }
        probs = session.run(["probs"], inputs)[0][0]
        probs = [float(p) for p in probs]
        vectors.append(
            {
                "utterance": utterance,
                "probs": probs,
                "argmax": int(np.argmax(probs)),
            }
        )
        print(f"{utterance!r:38} -> argmax={vectors[-1]['argmax']} "
              f"probs={[f'{p:.4f}' for p in probs]}")

    record = {
        "question": QUESTION,
        "options": OPTIONS,
        "ctx_len": CTX_LEN,
        "opt_len": OPT_LEN,
        "vectors": vectors,
        "generator": {
            "python_onnxruntime": ort.__version__,
            "note": (
                "vectors generated with ORT "
                + ort.__version__
                + " (ml_env via ~/v.sh); system ORT the C++ binary "
                "links may differ — int8 kernels can shift probs slightly"
            ),
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
