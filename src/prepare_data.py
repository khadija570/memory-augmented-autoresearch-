
from pathlib import Path
from itertools import islice
import hashlib
import json

import torch
from datasets import load_dataset


DATASET_NAME = "roneneldan/TinyStories"

N_TRAIN = 10_000
N_VALIDATION = 1_000
N_TEST = 1_000

EOS_TOKEN = 256
VOCAB_SIZE = 257

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def encode(text):
    return list(text.encode("utf-8"))


def tokenize_stories(stories):
    tokens = []

    for story in stories:
        tokens.extend(encode(story["text"]))
        tokens.append(EOS_TOKEN)

    return torch.tensor(tokens, dtype=torch.long)


def main():
    print("Loading TinyStories...")

    train_stream = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    validation_stream = load_dataset(
        DATASET_NAME,
        split="validation",
        streaming=True,
    )

    train_samples = list(
        islice(iter(train_stream), N_TRAIN)
    )

    validation_iterator = iter(validation_stream)

    validation_samples = list(
        islice(validation_iterator, N_VALIDATION)
    )

    test_samples = list(
        islice(validation_iterator, N_TEST)
    )

    # Check for overlapping stories
    train_texts = {s["text"] for s in train_samples}
    val_texts = {s["text"] for s in validation_samples}
    test_texts = {s["text"] for s in test_samples}

    overlaps = {
        "train_val": len(train_texts & val_texts),
        "train_test": len(train_texts & test_texts),
        "val_test": len(val_texts & test_texts),
    }

    print("Overlaps:", overlaps)

    if any(overlaps.values()):
        raise ValueError(
            "Dataset splits contain overlapping stories."
        )

    print("Tokenizing...")

    train_tokens = tokenize_stories(train_samples)
    val_tokens = tokenize_stories(validation_samples)
    test_tokens = tokenize_stories(test_samples)

    torch.save(train_tokens, DATA_DIR / "train.pt")
    torch.save(val_tokens, DATA_DIR / "val.pt")
    torch.save(test_tokens, DATA_DIR / "test.pt")

    metadata = {
        "dataset": DATASET_NAME,
        "train_stories": len(train_samples),
        "validation_stories": len(validation_samples),
        "test_stories": len(test_samples),
        "vocab_size": VOCAB_SIZE,
        "eos_token": EOS_TOKEN,
        "tokenizer": "utf-8 bytes",
        "token_counts": {
            "train": len(train_tokens),
            "validation": len(val_tokens),
            "test": len(test_tokens),
        },
    }

    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for name in ["train", "val", "test"]:
        file_path = DATA_DIR / f"{name}.pt"
        sha256 = hashlib.sha256(
            file_path.read_bytes()
        ).hexdigest()
        print(f"{name} SHA256: {sha256}")

    print("\nData preparation completed!")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
