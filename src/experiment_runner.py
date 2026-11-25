
import json
import math
import subprocess
import sys
import time
from pathlib import Path

SEARCH_SPACE = {
    "learning_rate": (0.0001, 0.003),
    "dropout": (0.0, 0.3),
    "weight_decay": (0.0, 0.1),
}

def validate_proposal(proposal):
    """Reject invalid or out-of-range experiments."""

    if not isinstance(proposal, dict):
        raise ValueError("Proposal must be a dictionary")

    if set(proposal) != set(SEARCH_SPACE):
        raise ValueError(
            "Proposal must contain exactly: "
            + ", ".join(SEARCH_SPACE)
        )

    validated = {}

    for name, (low, high) in SEARCH_SPACE.items():
        value = proposal[name]

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise ValueError(f"{name} must be numeric")

        value = float(value)

        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(
                f"{name} must be between {low} and {high}"
            )

        validated[name] = value

    return validated

def run_experiment(
    experiment_id,
    proposal,
    baseline_path="configs/baseline.json",
    output_root="results/autoresearch",
    max_steps=300,
    timeout_seconds=180,
):
    """Execute one controlled Mini-GPT experiment."""

    parameters = validate_proposal(proposal)

    with open(baseline_path, encoding="utf-8") as f:
        config = json.load(f)

    config.update(parameters)

    root = Path(output_root)
    experiment_dir = root / f"experiment_{experiment_id:03d}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config_path = experiment_dir / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        str(config_path),
        "--max_steps",
        str(max_steps),
        "--eval_interval",
        "50",
        "--eval_batches",
        "20",
        "--output_dir",
        str(experiment_dir),
    ]

    started = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

        elapsed = time.perf_counter() - started

        (experiment_dir / "stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (experiment_dir / "stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Training failed: {completed.stderr[-1500:]}"
            )

        history_path = experiment_dir / "history.json"

        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)

        best_record = min(
            history,
            key=lambda item: item["val_loss"],
        )

        result = {
            "experiment_id": experiment_id,
            "status": "success",
            "parameters": parameters,
            "best_val_loss": best_record["val_loss"],
            "best_step": best_record["step"],
            "elapsed_seconds": elapsed,
        }

    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        result = {
            "experiment_id": experiment_id,
            "status": "failed",
            "parameters": parameters,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }

    (experiment_dir / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result