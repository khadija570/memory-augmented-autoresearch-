
import json
import math
import time
from pathlib import Path

from src.research_agent import ResearchAgent
from src.experiment_runner import run_experiment


def save_history(history, path):
    """Persist experimental history safely."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(history, indent=4),
        encoding="utf-8"
    )

    temporary_path.replace(path)


def is_duplicate(parameters, history):
    """Check whether a configuration was already tested."""

    return any(
        record.get("parameters") == parameters
        for record in history
    )


def get_best_experiment(history):
    """Retrieve the best successful experiment."""

    successful = [
        record for record in history
        if record.get("status") == "success"
        and math.isfinite(record.get("best_val_loss", float("inf")))
    ]

    if not successful:
        raise ValueError("No valid baseline available")

    return min(
        successful,
        key=lambda record: record["best_val_loss"]
    )


def autonomous_research(
    max_experiments=5,
    max_proposal_attempts=3,
    history_path="results/autoresearch/experiment_history.json"
):

    with open("configs/baseline.json", encoding="utf-8") as f:
        baseline_config = json.load(f)

    history_path = Path(history_path)

    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    agent = ResearchAgent(
        model_name="gemini-3.8-flash"
    )

    best = get_best_experiment(history)

    next_id = max(
        record["experiment_id"] for record in history
    ) + 1

    started = time.perf_counter()
    completed_experiments = 0
    proposal_attempts = 0

    print("AUTONOMOUS RESEARCH STARTED")
    print("Initial best loss:", best["best_val_loss"])
    print("Experiment budget:", max_experiments)

    for iteration in range(max_experiments):

        print(f"\n{'=' * 50}")
        print(f"Research iteration {iteration + 1}")
        print(f"{'=' * 50}")

        proposal = None

        # Generate a valid, non-duplicate proposal
        for attempt in range(max_proposal_attempts):

            proposal_attempts += 1

            try:
                candidate = agent.propose(
                    history,
                    baseline_config
                )

                if is_duplicate(
                    candidate["parameters"],
                    history
                ):
                    print("Duplicate proposal rejected")
                    continue

                proposal = candidate
                break

            except Exception as error:
                print(
                    f"Proposal attempt {attempt + 1} failed:",
                    type(error).__name__
                )

                time.sleep(2)

        if proposal is None:
            print("No valid proposal obtained. Stopping safely.")
            break

        print("Hypothesis:", proposal["hypothesis"])
        print("Parameters:", proposal["parameters"])

        experiment_id = next_id + iteration

        # Execute the experiment
        try:
            result = run_experiment(
                experiment_id=experiment_id,
                proposal=proposal["parameters"],
                max_steps=300
            )

        except Exception as error:
            result = {
                "experiment_id": experiment_id,
                "status": "failed",
                "parameters": proposal["parameters"],
                "error": type(error).__name__
            }

        # Evaluate the result
        previous_best_loss = best["best_val_loss"]

        if (
            result["status"] == "success"
            and math.isfinite(
                result.get("best_val_loss", float("inf"))
            )
        ):

            new_loss = result["best_val_loss"]

            improvement_pct = (
                (previous_best_loss - new_loss)
                / previous_best_loss
            ) * 100

            decision = (
                "KEEP" if new_loss < previous_best_loss
                else "DISCARD"
            )

        else:
            decision = "FAILED"
            improvement_pct = None

        # Store experimental knowledge
        record = {
            **result,
            "hypothesis": proposal["hypothesis"],
            "reasoning": proposal["reasoning"],
            "decision": decision,
            "improvement_pct": improvement_pct,
            "agent_model": agent.model_name
        }

        history.append(record)

        if decision == "KEEP":
            best = record

        save_history(history, history_path)

        completed_experiments += 1

        print("Validation loss:", result.get("best_val_loss"))
        print("Decision:", decision)
        print("Current best loss:", best["best_val_loss"])

    elapsed = time.perf_counter() - started

    summary = {
        "completed_experiments": completed_experiments,
        "proposal_attempts": proposal_attempts,
        "best_experiment": best,
        "total_elapsed_seconds": elapsed
    }

    print("\nAUTONOMOUS RESEARCH FINISHED")
    print(json.dumps(summary, indent=4))

    return summary


if __name__ == "__main__":
    autonomous_research(max_experiments=5)
