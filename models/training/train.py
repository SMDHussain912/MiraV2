import json
import shutil

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "../tamev-base"

TRAIN_FILE = "train.jsonl"
VAL_FILE = "validation.jsonl"
TEST_FILE = "test.jsonl"

OUTPUT_DIR = "mira-tamev"

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

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 2e-5
MAX_CONTEXT_LENGTH = 128
MAX_OPTION_LENGTH = 64
SEED = 42

# Reproducibility (Phase 7): same seed -> same shuffle order -> same run.
import random

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(filename):
    data = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


train_data = load_jsonl(TRAIN_FILE)
val_data = load_jsonl(VAL_FILE)
test_data = load_jsonl(TEST_FILE)


print(f"Training examples:   {len(train_data)}")
print(f"Validation examples: {len(val_data)}")
print(f"Test examples:       {len(test_data)}")


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print(f"Device: {device}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading TAMEV...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

model.to(device)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# ============================================================
# TOKENIZATION
# ============================================================

def prepare_batch(batch):
    """
    Convert a batch of Mira examples into the exact
    tensors expected by TAMEV.forward().
    """

    contexts = []

    for item in batch:
        context = (
            QUESTION
            + "\nUser request: "
            + item["text"]
        )

        contexts.append(context)

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    ctx = tokenizer(
        contexts,
        padding="max_length",
        truncation=True,
        max_length=MAX_CONTEXT_LENGTH,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # Options
    # --------------------------------------------------------

    # Same candidate list for every example.
    #
    # tokenizer gives:
    #
    # [batch * options, sequence_length]
    #
    # Then we reshape it into:
    #
    # [batch, options, sequence_length]

    option_tokens = tokenizer(
        OPTIONS,
        padding="max_length",
        truncation=True,
        max_length=MAX_OPTION_LENGTH,
        return_tensors="pt",
    )

    batch_size = len(batch)
    num_options = len(OPTIONS)
    option_length = option_tokens["input_ids"].shape[1]

    opt_input_ids = option_tokens["input_ids"].unsqueeze(0)
    opt_attention_mask = option_tokens["attention_mask"].unsqueeze(0)

    opt_input_ids = opt_input_ids.expand(
        batch_size,
        num_options,
        option_length,
    ).contiguous()

    opt_attention_mask = opt_attention_mask.expand(
        batch_size,
        num_options,
        option_length,
    ).contiguous()

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = []

    for item in batch:
        labels.append(
            OPTIONS.index(item["label"])
        )

    labels = torch.tensor(
        labels,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    return (
        ctx["input_ids"].to(device),
        ctx["attention_mask"].to(device),
        opt_input_ids.to(device),
        opt_attention_mask.to(device),
        labels.to(device),
    )


# ============================================================
# FORWARD PASS
# ============================================================

def forward_batch(batch):
    (
        ctx_input_ids,
        ctx_attention_mask,
        opt_input_ids,
        opt_attention_mask,
        labels,
    ) = prepare_batch(batch)

    result = model(
        ctx_input_ids=ctx_input_ids,
        ctx_attention_mask=ctx_attention_mask,
        opt_input_ids=opt_input_ids,
        opt_attention_mask=opt_attention_mask,
        num_options=[len(OPTIONS)] * len(batch),
    )

    logits = result["logits"]

    loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
    )

    return loss, logits, labels


# ============================================================
# PREDICTION / ACCURACY
# ============================================================

def evaluate(dataset):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():

        for start in range(0, len(dataset), BATCH_SIZE):

            batch = dataset[
                start:start + BATCH_SIZE
            ]

            loss, logits, labels = forward_batch(batch)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += len(batch)

            total_loss += loss.item() * len(batch)

    accuracy = correct / total
    average_loss = total_loss / total

    return average_loss, accuracy


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# TRAINING
# ============================================================

print("\n==============================")
print("Starting training")
print("==============================\n")

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    # Shuffle training examples
    shuffled = train_data.copy()

    import random
    random.seed(SEED + epoch)
    random.shuffle(shuffled)

    total_loss = 0.0
    batches = 0

    for start in range(
        0,
        len(shuffled),
        BATCH_SIZE,
    ):

        batch = shuffled[
            start:start + BATCH_SIZE
        ]

        optimizer.zero_grad()

        loss, logits, labels = forward_batch(batch)

        loss.backward()

        # Prevent unusually large gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        batches += 1

    train_loss = total_loss / batches

    val_loss, val_accuracy = evaluate(
        val_data
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2%}"
    )

    # Save best model
    if val_accuracy >= best_val_accuracy:

        best_val_accuracy = val_accuracy

        print("  -> Saving best model")

        model.save_pretrained(
            OUTPUT_DIR
        )

        tokenizer.save_pretrained(
            OUTPUT_DIR
        )


# ============================================================
# TEST
# ============================================================

print("\n==============================")
print("Final test")
print("==============================")

test_loss, test_accuracy = evaluate(
    test_data
)

print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")


# ============================================================
# COPY CUSTOM MODEL CODE
# ============================================================

print("\nCopying TAMEV custom code...")

shutil.copy(
    "../tamev-base/configuration_tamev.py",
    OUTPUT_DIR,
)

shutil.copy(
    "../tamev-base/modeling_tamev.py",
    OUTPUT_DIR,
)

print("\nTraining complete!")
print(f"Model saved to: {OUTPUT_DIR}")
