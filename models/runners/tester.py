from transformers import AutoTokenizer, AutoModel

model_id = "Tamkimd/tamev-nano-tinybert"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True
)

result = model.predict_decision(
    state="The user wants to open Firefox.",
    question="What should Mira do?",
    options=[
        "open_application",
        "search_web",
        "read_screen",
        "conversation"
    ],
    tokenizer=tokenizer
)

print("Selected:", result["best_option"])
print("Confidence:", result["confidence"])
print("Probabilities:", result["probabilities"])
