
import tempfile
from pathlib import Path

from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory

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


def test_semantic_memory():

    def make_experiment(
        experiment_id,
        lr,
        dropout,
        wd,
        loss,
        valid=True,
        protocol_id="test_protocol_v1"
    ):

        return {
            "experiment_id": experiment_id,
            "status": "success",
            "parameters": {
                "learning_rate": lr,
                "dropout": dropout,
                "weight_decay": wd
            },
            "best_val_loss": loss,
            "valid": valid,
            "protocol_id": protocol_id
        }

    # Test fixtures based on pilot observations
    exp_0 = make_experiment(
        0, 0.001, 0.1, 0.01, 2.002828
    )

    exp_3 = make_experiment(
        3, 0.001, 0.0, 0.01, 1.903852
    )

    exp_4 = make_experiment(
        4, 0.001, 0.0, 0.0, 1.903852,
        valid=False
    )

    exp_7 = make_experiment(
        7, 0.0007, 0.0, 0.01, 1.894414
    )

    memory = SemanticMemory()

    # Test 1: Controlled comparison
    finding = memory.compare_experiments(
        exp_0, exp_3
    )

    assert finding is not None
    assert finding["parameter"] == "dropout"

    print("PASS: Controlled comparison")

    # Test 2: Correct effect classification
    assert finding["effect"] == "degraded"

    print("PASS: Effect classification")

    # Test 3: Reject confounded comparisons
    assert memory.compare_experiments(
        exp_0, exp_7
    ) is None

    print("PASS: Confounded comparison rejected")

    # Test 4: Reject invalid evidence
    assert memory.compare_experiments(
        exp_3, exp_4
    ) is None

    print("PASS: Invalid evidence rejected")

    # Test 5: Reject incompatible protocols
    different_protocol = {
        **exp_3,
        "protocol_id": "different_protocol"
    }

    assert memory.compare_experiments(
        exp_0, different_protocol
    ) is None

    print("PASS: Incompatible protocol rejected")

    # Test 6: Automatic knowledge extraction
    findings = memory.extract_findings([
        exp_0,
        exp_3,
        exp_4,
        exp_7
    ])

    assert len(findings) == 2

    print("PASS: Automatic knowledge extraction")

    # Test 7: Parameter-based retrieval
    dropout_findings = memory.get_by_parameter(
        "dropout"
    )

    assert len(dropout_findings) == 1

    print("PASS: Parameter-based retrieval")

    # Test 8: Effect-based retrieval
    strongest = memory.get_strongest(k=1)

    assert len(strongest) == 1
    assert strongest[0]["parameter"] == "dropout"

    print("PASS: Effect-based retrieval")

    print("\nEXTRACTED FINDINGS:")

    for finding in findings:

        print(finding["observation"])

        print(
            "Supporting experiments:",
            finding["supporting_experiments"]
        )

        print(
            "Improvement:",
            round(finding["improvement_pct"], 4),
            "%"
        )

        print()

    print("ALL SEMANTIC MEMORY TESTS PASSED")


if __name__ == "__main__":

    test_episodic_memory()
    test_semantic_memory()
