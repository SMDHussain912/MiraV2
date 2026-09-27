import json
from sklearn.model_selection import train_test_split
from dataset import data


# 80% training, 20% temporary
train_data, temp_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=[item["label"] for item in data]
)

# Split remaining 20% into:
# 10% validation
# 10% test
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=42,
    stratify=[item["label"] for item in temp_data]
)


def save_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")


save_jsonl("train.jsonl", train_data)
save_jsonl("validation.jsonl", val_data)
save_jsonl("test.jsonl", test_data)


print("Training:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))
