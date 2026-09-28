"""Per-intent metrics for the TAMEV intent classifier (Phase 7).

Pure Python (no sklearn) so train.py, eval_model.py and the export checks all
report the same numbers. A single label vocabulary shared with train.py keeps
the option order of the confusion matrix identical to the model's option order.
"""

INTENTS = [
    "open_application",
    "search_web",
    "read_screen",
    "type_text",
    "file_operation",
    "system_control",
    "conversation",
]


def confusion_matrix(gold, pred, labels=INTENTS):
    """Row = gold label, column = predicted label, both in option order."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for gold_label, pred_label in zip(gold, pred):
        matrix[index[gold_label]][index[pred_label]] += 1
    return matrix


def per_intent_metrics(matrix, labels=INTENTS):
    """Precision / recall / F1 / support per intent from a confusion matrix."""
    report = {}
    for i, label in enumerate(labels):
        true_positive = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator else 0.0
        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "predicted": predicted,
        }
    return report


def summarize(gold, pred, labels=INTENTS):
    """Accuracy, macro-F1 and balanced accuracy plus the per-intent breakdown."""
    matrix = confusion_matrix(gold, pred, labels)
    report = per_intent_metrics(matrix, labels)
    total = len(gold)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    f1_scores = [report[label]["f1"] for label in labels if report[label]["support"]]
    recalls = [report[label]["recall"] for label in labels if report[label]["support"]]
    return {
        "examples": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "balanced_accuracy": (
            round(sum(recalls) / len(recalls), 4) if recalls else 0.0
        ),
        "labels": list(labels),
        "confusion_matrix": matrix,
        "per_intent": report,
    }


def format_report(summary, title=""):
    """Human-readable table + confusion matrix for stdout."""
    labels = summary["labels"]
    lines = []
    if title:
        lines.append(title)
    lines.append(
        f"  examples={summary['examples']}  accuracy={summary['accuracy']:.4f}  "
        f"macro_f1={summary['macro_f1']:.4f}  "
        f"balanced_accuracy={summary['balanced_accuracy']:.4f}"
    )
    lines.append(f"  {'intent':<18}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
    for label in labels:
        row = summary["per_intent"][label]
        lines.append(
            f"  {label:<18}{row['precision']:>7.3f}{row['recall']:>8.3f}"
            f"{row['f1']:>7.3f}{row['support']:>9}"
        )
    width = max(len(label) for label in labels) + 2
    header = " " * width + "".join(f"{label[:6]:>7}" for label in labels)
    lines.append("  confusion matrix (rows = gold, columns = predicted)")
    lines.append("  " + header)
    for i, label in enumerate(labels):
        cells = "".join(f"{value:>7}" for value in summary["confusion_matrix"][i])
        lines.append(f"  {label:<{width}}{cells}")
    return "\n".join(lines)
