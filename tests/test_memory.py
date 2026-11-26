
import tempfile
from pathlib import Path

from src.memory.episodic import EpisodicMemory


def test_episodic_memory():

    with tempfile.TemporaryDirectory() as directory:

        memory_path = (
            Path(directory) / "episodic_memory.json"
        )

        memory = EpisodicMemory(memory_path)

        # Experiment 0: baseline
        memory.add_experiment({
            "experiment_id": 0,
            "status": "success",
            "parameters": {
                "learning_rate": 0.001,
                "dropout": 0.1,
                "weight_decay": 0.01
            },
            "best_val_loss": 2.002828,
            "decision": "KEEP"
        })

        # Experiment 3: improvement
        memory.add_experiment({
            "experiment_id": 3,
            "status": "success",
            "parameters": {
                "learning_rate": 0.001,
                "dropout": 0.0,
                "weight_decay": 0.01
            },
            "best_val_loss": 1.903852,
            "decision": "KEEP"
        })

        # Experiment 4: invalid execution
        memory.add_experiment({
            "experiment_id": 4,
            "status": "success",
            "parameters": {
                "learning_rate": 0.001,
                "dropout": 0.0,
                "weight_decay": 0.0
            },
            "best_val_loss": 1.903852,
            "decision": "DISCARD"
        })

        memory.mark_invalid(
            4,
            "Weight decay was ignored by AdamW"
        )

        # Experiment 7: best configuration
        memory.add_experiment({
            "experiment_id": 7,
            "status": "success",
            "parameters": {
                "learning_rate": 0.0007,
                "dropout": 0.0,
                "weight_decay": 0.01
            },
            "best_val_loss": 1.894414,
            "decision": "KEEP"
        })

        # Test 1: Storage
        assert len(memory) == 4
        print("PASS: Experiment storage")

        # Test 2: Individual retrieval
        assert memory.get_experiment(3) is not None
        print("PASS: Individual retrieval")

        # Test 3: Invalid evidence exclusion
        assert len(memory.get_successful()) == 3
        print("PASS: Invalid evidence exclusion")

        # Test 4: Best experiment retrieval
        best = memory.get_best(k=1)[0]

        assert best["experiment_id"] == 7
        print("PASS: Best experiment retrieval")

        # Test 5: Recent experiment retrieval
        recent = memory.get_recent(k=2)

        assert len(recent) == 2
        assert recent[-1]["experiment_id"] == 7
        print("PASS: Recent experiment retrieval")

        # Test 6: Duplicate rejection
        try:
            memory.add_experiment(
                memory.get_experiment(7)
            )
        except ValueError:
            print("PASS: Duplicate rejection")
        else:
            raise AssertionError(
                "Duplicate was not rejected"
            )

        # Test 7: Persistence
        restored = EpisodicMemory(memory_path)

        assert restored.get_all() == memory.get_all()
        print("PASS: Memory persistence")

        print("\nALL EPISODIC MEMORY TESTS PASSED")


if __name__ == "__main__":

    test_episodic_memory()
