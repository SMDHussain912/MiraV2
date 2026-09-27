import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# ==================================================
# Paths
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_int8.onnx"
)


# ==================================================
# Tokenizer
# ==================================================

MODEL_ID = "Tamkimd/tamev-nano-tinybert"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)


# ==================================================
# Load ONNX model
# ==================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)


# ==================================================
# Mira decision
# ==================================================

state = "Open Firefox"

question = "What should Mira do?"

options = [
    "open_application",
    "search_web",
    "read_screen",
    "conversation"
]


# ==================================================
# Tokenize context
# ==================================================

context_text = state + " " + question

ctx = tokenizer(
    context_text,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="np"
)


# ==================================================
# Tokenize options
# ==================================================

option_ids = []
option_masks = []

for option in options:

    encoded = tokenizer(
        option,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="np"
    )

    option_ids.append(
        encoded["input_ids"][0]
    )

    option_masks.append(
        encoded["attention_mask"][0]
    )


# ==================================================
# Convert options to NumPy arrays
# ==================================================

opt_input_ids = np.array(
    option_ids,
    dtype=np.int64
)

opt_attention_mask = np.array(
    option_masks,
    dtype=np.int64
)


# ==================================================
# Add batch dimension
# ==================================================

ctx_input_ids = ctx["input_ids"].astype(
    np.int64
)

ctx_attention_mask = ctx["attention_mask"].astype(
    np.int64
)

opt_input_ids = np.expand_dims(
    opt_input_ids,
    axis=0
)

opt_attention_mask = np.expand_dims(
    opt_attention_mask,
    axis=0
)


# ==================================================
# Prepare ONNX inputs
# ==================================================

inputs = {

    "ctx_input_ids":
        ctx_input_ids,

    "ctx_attention_mask":
        ctx_attention_mask,

    "opt_input_ids":
        opt_input_ids,

    "opt_attention_mask":
        opt_attention_mask
}


# ==================================================
# Run inference
# ==================================================

outputs = session.run(
    ["logits", "probs"],
    inputs
)


logits = outputs[0]
probs = outputs[1]


# ==================================================
# Get result
# ==================================================

probabilities = probs[0]

best_index = int(
    np.argmax(probabilities)
)

best_option = options[best_index]


# ==================================================
# Display
# ==================================================

print()
print("User:", state)
print()

print("Decisions:")

for option, probability in zip(
    options,
    probabilities
):

    print(
        f"{option:20s} "
        f"{float(probability):.4f}"
    )

print()

print("Selected:", best_option)

print(
    "Confidence:",
    float(probabilities[best_index])
)
