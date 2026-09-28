#!/usr/bin/env python3
"""Build balanced, leakage-safe train/validation/test splits for Phase 7.

Offline dev step only (runs in ~/ml_env, never in the C++ runtime).

    source ~/ml_env/bin/activate
    cd models/training && python3 build_splits.py

Inputs (see models/training/datasheets/tamev-intent-classifier.md for the
contract):
  * dataset.py                      70 curated rows, 7 intents (smoke corpus)
  * datasets/open_application.jsonl 1165 generated rows, template heavy
  * datasets/search_web.jsonl       1016 generated rows, template heavy
  * datasets/<intent>_v2.jsonl      gap-closing hand-written rows for the five
                                    intents the generated corpora never covered
                                    (found by the Phase 7 baseline error analysis)
  * datasets/hard_negatives_v2.jsonl boundary cases for the confusions the
                                    baseline confusion matrix showed

Rules implemented here, in order:
  1. schema validation (non-empty text, label in the seven-option catalog);
  2. exact dedup: case-fold + whitespace normalise + punctuation-insensitive,
     curated/hand-written rows win over generated rows;
  3. near-duplicate control: every row gets a group id = (intent, template
     skeleton) where low-frequency "entity" tokens are wildcarded, so template
     families and paraphrase clusters are one indivisible group; oversized
     groups are subdivided by entity signature so the same app/query string can
     never appear in two splits;
  4. class balance: each intent is capped at TARGET_PER_INTENT whole groups so
     the two generated intents cannot swamp the other five;
  5. stratified 80/10/10 split assigned by GROUP, never by row;
  6. leakage audit: exact-normalised text, group id and entity token
     intersections across the three splits are reported; exact and group
     intersections must be zero or the script fails.

Outputs: corpus.jsonl, train.jsonl, validation.jsonl, test.jsonl and
split_report.json (integrity + balance + leakage report).
"""
import collections
import hashlib
import json
import os
import random
import re
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEED = 42
TARGET_PER_INTENT = 90          # balance cap, in rows, per intent
SKELETON_MAX_GROUP = 24         # split bigger template groups by entity hash
RARE_TOKEN_DF = 5               # tokens in < 5 rows of an intent = entity slot
RATIOS = (("train", 0.80), ("val", 0.10), ("test", 0.10))
INTENTS = ["open_application", "search_web", "read_screen", "type_text",
           "file_operation", "system_control", "conversation"]
GENERATED_FILES = ["open_application.jsonl", "search_web.jsonl"]
AUGMENTED_FILES = ["read_screen_v2.jsonl", "type_text_v2.jsonl",
                   "file_operation_v2.jsonl", "system_control_v2.jsonl",
                   "conversation_v2.jsonl", "hard_negatives_v2.jsonl"]
STOPWORDS = set("""a an the my our me you it this that these those for of to in
on at and or then please can could would will you're im i is are do does
you can""".split())


def norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def norm_nopunct(text):
    return norm(text).translate(str.maketrans("", "", string.punctuation))


def tokens(text):
    """Content tokens: punctuation-insensitive, lower-cased, stop-words removed."""
    cleaned = norm_nopunct(text)
    return [t for t in cleaned.split() if t not in STOPWORDS]


def load_jsonl(path, source):
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
            if label not in INTENTS:
                raise SystemExit(f"{path}:{number}: invalid label {label!r}")
            rows.append({"text": text.strip(), "label": label, "source": source})
    return rows


def load_all():
    from dataset import data as curated
    rows = [{"text": r["text"].strip(), "label": r["label"], "source": "dataset.py"}
            for r in curated]
    for name in GENERATED_FILES:
        rows += load_jsonl(os.path.join(HERE, "datasets", name), "generated/" + name)
    for name in AUGMENTED_FILES:
        rows += load_jsonl(os.path.join(HERE, "datasets", name), "handwritten/" + name)
    return rows



def dedup(rows):
    """Exact dedup on punctuation-insensitive normalised text.

    Priority: curated (dataset.py) > handwritten > generated, so the clean
    source always wins the collision.  Returns kept rows plus audit counts.
    """
    rank_of = lambda src: (0 if src == "dataset.py"
                           else 1 if src.startswith("handwritten") else 2)
    best, dropped = {}, 0
    for row in rows:
        key = norm_nopunct(row["text"])
        if key in best:
            dropped += 1
            if rank_of(row["source"]) < rank_of(best[key]["source"]):
                best[key] = row
        else:
            best[key] = row
    return list(best.values()), dropped


def build_groups(rows):
    """Assign each row a group id (intent, template skeleton[, entity bucket]).

    The skeleton wildcards low-frequency entity tokens (app names, file names,
    query strings) so a template family stays in one split.  Groups larger than
    SKELETON_MAX_GROUP are bucketed by hash of their entity signature, which
    keeps identical entity strings from leaking across splits.
    """
    per_intent = collections.defaultdict(list)
    for row in rows:
        per_intent[row["label"]].append(row)

    for label, intent_rows in per_intent.items():
        df = collections.Counter()
        for row in intent_rows:
            df.update(set(tokens(row["text"])))
        for row in intent_rows:
            row["_entity"] = tuple(sorted(t for t in tokens(row["text"])
                                          if df[t] < RARE_TOKEN_DF))
            skeleton = " ".join("<E>" if df[t] < RARE_TOKEN_DF else t
                                for t in tokens(row["text"]))
            row["_group"] = (label, skeleton)
        groups = collections.defaultdict(list)
        for row in intent_rows:
            groups[row["_group"]].append(row)
        for (lab, skeleton), members in groups.items():
            if len(members) <= SKELETON_MAX_GROUP:
                for row in members:
                    row["group_id"] = f"{lab}|{skeleton}"
            else:
                buckets = (len(members) + SKELETON_MAX_GROUP - 1) // SKELETON_MAX_GROUP
                for row in members:
                    digest = hashlib.sha1("|".join(row["_entity"]).encode()).hexdigest()[:6]
                    row["group_id"] = (f"{lab}|{skeleton}"
                                       f"#{int(digest, 16) % buckets}")
    return rows


def balance_and_split(rows, seed=SEED):
    """Cap each intent to TARGET_PER_INTENT whole groups, then 80/10/10 by group."""
    rng = random.Random(seed)
    by_intent = collections.defaultdict(list)
    for row in rows:
        by_intent[row["label"]].append(row)

    kept, balance = [], {}
    for label in INTENTS:
        intent_rows = by_intent.get(label, [])
        groups = collections.defaultdict(list)
        for row in intent_rows:
            groups[row["group_id"]].append(row)
        keys = sorted(groups)
        rng.shuffle(keys)
        keys.sort(key=lambda k: len(groups[k]))  # small groups first -> finer balance
        chosen, total = [], 0
        for key in keys:
            if total + len(groups[key]) > TARGET_PER_INTENT and total >= TARGET_PER_INTENT // 2:
                continue
            chosen.append(key)
            total += len(groups[key])
        balance[label] = {"rows_before_cap": len(intent_rows),
                          "groups": len(groups), "rows_kept": total,
                          "groups_kept": len(chosen)}
        for key in chosen:
            kept.extend(groups[key])

    # stratified group-level 80/10/10
    splits = {"train": [], "val": [], "test": []}
    per_intent_split = collections.defaultdict(lambda: collections.Counter())
    for label in INTENTS:
        groups = collections.defaultdict(list)
        for row in kept:
            if row["label"] == label:
                groups[row["group_id"]].append(row)
        keys = sorted(groups)
        rng.shuffle(keys)
        # greedily assign whole groups: next group goes to whichever split is
        # furthest below its target ratio (remainder lands in train)
        want = {name: ratio * sum(len(groups[k]) for k in keys)
                for name, ratio in RATIOS}
        fill = {"train": 0, "val": 0, "test": 0}
        for key in sorted(keys, key=lambda k: len(groups[k]), reverse=True):
            size = len(groups[key])
            candidates = [(name, (fill[name] + size > want[name],
                                  fill[name] - want[name]))
                          for name, _ in RATIOS]
            candidates.sort(key=lambda c: (c[1][0], c[1][1]))
            target = candidates[0][0]
            fill[target] += size
            per_intent_split[label][target] += size
            splits[target].extend(groups[key])
    return kept, splits, balance, per_intent_split



def leakage_audit(splits):
    """Cross-split intersections that must be zero, plus advisory warnings."""
    keys = {name: {norm_nopunct(r["text"]) for r in rows}
            for name, rows in splits.items()}
    groups = {name: {r["group_id"] for r in rows}
              for name, rows in splits.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    exact = {f"{a}^{b}": len(keys[a] & keys[b]) for a, b in pairs}
    group = {f"{a}^{b}": len(groups[a] & groups[b]) for a, b in pairs}

    # advisory: entity tokens (rare within an intent) shared across splits.
    # Group bucketing makes this small but not provably zero for entities that
    # occur under different skeletons; reported, never silently ignored.
    ent = {}
    for a, b in pairs:
        ea = collections.defaultdict(set)
        eb = collections.defaultdict(set)
        for row in splits[a]:
            ea[row["label"]].update(row["_entity"])
        for row in splits[b]:
            eb[row["label"]].update(row["_entity"])
        ent[f"{a}^{b}"] = {label: len(ea[label] & eb[label])
                           for label in INTENTS}
    hard_fail = {k: v for k, v in {**exact, **group}.items() if v}
    return {"exact_norm_text_overlap": exact, "group_id_overlap": group,
            "entity_token_overlap_advisory": ent}, hard_fail


def main():
    raw = load_all()
    kept, dropped_dupes = dedup(raw)
    kept = build_groups(kept)
    corpus, splits, balance, per_intent = balance_and_split(kept)

    issues, hard_fail = leakage_audit(splits)

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(
                    {"text": row["text"], "label": row["label"],
                     "group_id": row["group_id"],
                     "source_id": row["source"]}, ensure_ascii=False) + "\n")

    dump(os.path.join(HERE, "corpus.jsonl"), corpus)
    dump(os.path.join(HERE, "train.jsonl"), splits["train"])
    dump(os.path.join(HERE, "validation.jsonl"), splits["val"])
    dump(os.path.join(HERE, "test.jsonl"), splits["test"])

    def counts(rows):
        c = collections.Counter(r["label"] for r in rows)
        return {label: c.get(label, 0) for label in INTENTS}

    report = {
        "seed": SEED,
        "target_per_intent": TARGET_PER_INTENT,
        "raw_rows": len(raw),
        "dropped_exact_duplicates": dropped_dupes,
        "corpus_rows": len(corpus),
        "corpus_label_counts": counts(corpus),
        "split_rows": {k: len(v) for k, v in splits.items()},
        "split_label_counts": {k: counts(v) for k, v in splits.items()},
        "balance": balance,
        "per_intent_split_rows": {k: dict(v) for k, v in per_intent.items()},
        "leakage_audit": issues,
        "status": "FAIL" if hard_fail else "OK",
        "hard_failures": hard_fail,
        "sources": sorted({r["source"] for r in raw}),
    }
    with open(os.path.join(HERE, "split_report.json"), "w", encoding="utf-8") as h:
        json.dump(report, h, indent=2, ensure_ascii=False)

    print(f"raw={len(raw)} dedup_dropped={dropped_dupes} corpus={len(corpus)}")
    for name, rows in splits.items():
        print(f"  {name:10s} {len(rows):4d}  {counts(rows)}")
    print("leakage exact:", json.dumps(issues["exact_norm_text_overlap"]))
    print("leakage group:", json.dumps(issues["group_id_overlap"]))
    print("status:", report["status"], hard_fail or "")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
