
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.model import MiniGPT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_batch(data, batch_size, block_size, device):
    """Create random next-token prediction examples."""

    positions = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
    )

    offsets = torch.arange(block_size)

    x = data[positions[:, None] + offsets]
    y = data[positions[:, None] + offsets + 1]

    return x.to(device), y.to(device)



@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    batch_size,
    block_size,
    device,
    eval_batches=20,
):
    model.eval()
    results = {}

    for split, data in [
        ("train", train_data),
        ("val", val_data),
    ]:
        # Independent random generator for evaluation
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            2027 if split == "val" else 2028
        )

        losses = []

        for _ in range(eval_batches):
            positions = torch.randint(
                0,
                len(data) - block_size,
                (batch_size,),
                generator=generator,
            )

            offsets = torch.arange(block_size)

            x = data[positions[:, None] + offsets]
            y = data[positions[:, None] + offsets + 1]

            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)
            losses.append(loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()
    return results

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output_dir", default="results/baseline")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    set_seed(config["seed"])

    # Request deterministic operations where supported
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_data = torch.load(
        "data/train.pt",
        map_location="cpu",
        weights_only=True,
    )

    val_data = torch.load(
        "data/val.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = MiniGPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config.get("weight_decay", 0.01),
    )
    print("Optimizer configuration:")
    print("Learning rate:", optimizer.param_groups[0]["lr"])
    print("Weight decay:", optimizer.param_groups[0]["weight_decay"])

    batch_size = config["batch_size"]
    block_size = config["block_size"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float("inf")

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    print("Starting training...")

    for step in range(args.max_steps + 1):

        if step % args.eval_interval == 0 or step == args.max_steps:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                batch_size,
                block_size,
                device,
                args.eval_batches,
            )

            if device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time

            record = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "elapsed_seconds": elapsed,
            }

            history.append(record)

            print(
                f"Step {step:4d} | "
                f"Train: {losses['train']:.4f} | "
                f"Val: {losses['val']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "step": step,
                        "val_loss": best_val_loss,
                    },
                    output_dir / "best_model.pt",
                )

        if step == args.max_steps:
            break

        x, y = get_batch(
            train_data,
            batch_size,
            block_size,
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    if device == "cuda":
        torch.cuda.synchronize()

    total_seconds = time.perf_counter() - start_time

    print(f"Total experiment time: {total_seconds:.2f} seconds")
    print(f"Average time per step: {total_seconds / args.max_steps:.4f} seconds")

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
