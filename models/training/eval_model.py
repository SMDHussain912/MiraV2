#!/usr/bin/env python3
"""Evaluate a TAMEV checkpoint on a JSONL split with per-intent metrics (Phase 7).

Offline dev step only — runs inside ~/ml_env, never in the C++ runtime.

    source ~/ml_env/bin/activate
    cd models/training
    python3 eval_model.py --model mira-tamev --data test.jsonl \
        --report reports/eval_baseline_test.json --show-errors 25

Uses the canonical runtime contract (see models/parity/README.md): the context is
"Which action should Mira perform?\\nUser request: <text>" padded to ctx_len=128
and the seven options padded to opt_len=64, exactly as the C++ classifier
encodes them, so the numbers here describe the exported artifact's behaviour.
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTION = "Which action should Mira perform?"
OPTIONS = metrics.INTENTS
CTX_LEN = 128
OPT_LEN = 64


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text, label = row.get("text"), row.get("label")
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(f"{path}:{number}: missing or empty 'text'")
            if label not in OPTIONS:
                raise SystemExit(f"{path}:{number}: invalid label {label!r}")
            rows.append({"text": text, "label": label})
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def encode(tokenizer, texts, max_length):
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def option_tensors(tokenizer):
    ids, mask = encode(tokenizer, OPTIONS, OPT_LEN)
    return ids.unsqueeze(0), mask.unsqueeze(0)


def predict(model, tokenizer, rows, batch_size=16):
    """Return (gold, predicted, max softmax probability) over the whole split."""
    opt_ids, opt_mask = option_tensors(tokenizer)
    gold, predicted, confidences = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            contexts = [QUESTION + "\nUser request: " + row["text"] for row in chunk]
            ctx_ids, ctx_mask = encode(tokenizer, contexts, CTX_LEN)
            size = len(chunk)
            result = model(
                ctx_input_ids=ctx_ids,
                ctx_attention_mask=ctx_mask,
                opt_input_ids=opt_ids.expand(size, -1, -1).contiguous(),
                opt_attention_mask=opt_mask.expand(size, -1, -1).contiguous(),
            )
            probs = result["probs"]
            best = probs.argmax(dim=1)
            for i, row in enumerate(chunk):
                gold.append(row["label"])
                predicted.append(OPTIONS[int(best[i])])
                confidences.append(float(probs[i, best[i]]))
    return gold, predicted, confidences



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.join(HERE, "mira-tamev"),
                        help="checkpoint directory to evaluate")
    parser.add_argument("--data", default=os.path.join(HERE, "test.jsonl"),
                        help="JSONL split to evaluate")
    parser.add_argument("--report", default=None,
                        help="write the full metrics JSON to this path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--show-errors", type=int, default=20,
                        help="print at most this many misclassified utterances")
    args = parser.parse_args()

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True,
                                      trust_remote_code=True)
    model.eval()

    rows = load_rows(args.data)
    gold, predicted, confidences = predict(model, tokenizer, rows, args.batch_size)
    summary = metrics.summarize(gold, predicted)

    errors = [
        {
            "text": row["text"],
            "gold": row["label"],
            "predicted": predicted[i],
            "confidence": round(confidences[i], 4),
        }
        for i, row in enumerate(rows)
        if predicted[i] != gold[i]
    ]
    error_pairs = {}
    for error in errors:
        key = f"{error['gold']} -> {error['predicted']}"
        error_pairs[key] = error_pairs.get(key, 0) + 1

    print(f"model: {args.model}\ndata:  {args.data}")
    print(metrics.format_report(summary, title="== per-intent metrics =="))
    print(f"  errors: {len(errors)} / {len(rows)}")
    if error_pairs:
        print("  top confusions:")
        for pair, count in sorted(error_pairs.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {pair:<42}{count}")
    if errors and args.show_errors:
        print("  misclassified utterances:")
        for error in errors[: args.show_errors]:
            print(f"    [{error['gold']} != {error['predicted']} "
                  f"p={error['confidence']:.3f}] {error['text']}")

    if args.report:
        report = {
            "model": os.path.relpath(os.path.abspath(args.model), HERE),
            "data": os.path.relpath(os.path.abspath(args.data), HERE),
            "question": QUESTION,
            "options": OPTIONS,
            "ctx_len": CTX_LEN,
            "opt_len": OPT_LEN,
            "metrics": summary,
            "error_count": len(errors),
            "error_pairs": error_pairs,
            "errors": errors,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
            handle.write("\n")
        print(f"  wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
