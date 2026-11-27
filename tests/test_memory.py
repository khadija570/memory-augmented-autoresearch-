
import tempfile
from pathlib import Path

from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.memory.failure import FailureMemory
from src.memory.retrieval import MemoryRetriever
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

def test_failure_memory():

    memory = FailureMemory()

    # Test 1: Scientific failure
    exp_5 = {
        "experiment_id": 5,
        "status": "success",
        "parameters": {
            "learning_rate": 0.0012,
            "dropout": 0.0,
            "weight_decay": 0.01
        },
        "best_val_loss": 2.004559648036957,
        "valid": True
    }

    failure = memory.add_experiment(
        exp_5,
        reference_loss=1.9038520574569702
    )

    assert failure["failure_type"] == "scientific_failure"
    assert failure["scientific_evidence"] is True

    print("PASS: Scientific failure classification")

    # Test 2: No improvement
    exp_equal = {
        **exp_5,
        "experiment_id": 8,
        "best_val_loss": 1.9038520574569702
    }

    failure = memory.add_experiment(
        exp_equal,
        reference_loss=1.9038520574569702
    )

    assert failure["failure_type"] == "no_improvement"

    print("PASS: No-improvement classification")

    # Test 3: Successful improvement
    exp_7 = {
        "experiment_id": 7,
        "status": "success",
        "parameters": {
            "learning_rate": 0.0007,
            "dropout": 0.0,
            "weight_decay": 0.01
        },
        "best_val_loss": 1.894414496421814,
        "valid": True
    }

    failure = memory.add_experiment(
        exp_7,
        reference_loss=1.8996867656707763
    )

    assert failure is None

    print("PASS: Successful improvement excluded")

    # Test 4: Invalid experimental evidence
    exp_4 = {
        "experiment_id": 4,
        "status": "success",
        "parameters": {
            "learning_rate": 0.001,
            "dropout": 0.0,
            "weight_decay": 0.0
        },
        "best_val_loss": 1.9038520574569702,
        "valid": False,
        "invalid_reason": (
            "Weight decay was ignored by AdamW"
        )
    }

    failure = memory.add_experiment(exp_4)

    assert failure["failure_type"] == "invalid_experiment"
    assert failure["scientific_evidence"] is False

    print("PASS: Invalid experiment classification")

    # Test 5: Operational training failure
    failed_experiment = {
        "experiment_id": 9,
        "status": "failed",
        "parameters": {},
        "error": "Training timeout"
    }

    failure = memory.add_experiment(
        failed_experiment
    )

    assert failure["failure_type"] == "operational_failure"
    assert failure["scientific_evidence"] is False

    print("PASS: Operational failure classification")

    # Test 6: Gemini API failure
    failure = memory.add_proposal_failure(
        reason="503 UNAVAILABLE",
        attempt=1,
        operational=True
    )

    assert failure["failure_type"] == "operational_failure"
    assert failure["scientific_evidence"] is False

    print("PASS: API failure classification")

    # Test 7: Invalid proposal
    failure = memory.add_proposal_failure(
        reason="Learning rate outside search space",
        parameters={"learning_rate": 10},
        attempt=2
    )

    assert failure["failure_type"] == "proposal_failure"

    print("PASS: Proposal failure classification")

    # Test 8: Scientific evidence retrieval
    scientific = memory.get_scientific_failures()

    assert len(scientific) == 2

    print("PASS: Scientific evidence retrieval")

    # Test 9: Failure summary
    summary = memory.get_summary()

    assert summary["scientific_failure"] == 1
    assert summary["no_improvement"] == 1
    assert summary["invalid_experiment"] == 1
    assert summary["operational_failure"] == 2
    assert summary["proposal_failure"] == 1

    assert len(memory) == 6

    print("PASS: Failure summary")

    print("\nFAILURE MEMORY SUMMARY:")

    for category, count in summary.items():

        print(f"{category}: {count}")

    print("\nALL FAILURE MEMORY TESTS PASSED")


def test_memory_retrieval():

    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    failure = FailureMemory()

    protocol_id = "test_protocol_v1"

    def make_experiment(
        experiment_id,
        lr,
        dropout,
        wd,
        loss,
        valid=True,
        protocol=protocol_id
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
            "protocol_id": protocol
        }

    experiments = [

        make_experiment(
            0, 0.001, 0.1, 0.01, 2.002828
        ),

        make_experiment(
            3, 0.001, 0.0, 0.01, 1.903852
        ),

        make_experiment(
            4, 0.001, 0.0, 0.0, 1.903852,
            valid=False
        ),

        make_experiment(
            5, 0.0012, 0.0, 0.01, 2.004560
        ),

        make_experiment(
            6, 0.0008, 0.0, 0.01, 1.899687
        ),

        make_experiment(
            7, 0.0007, 0.0, 0.01, 1.894414
        ),

        # Incompatible synthetic experiment
        make_experiment(
            99, 0.001, 0.0, 0.01, 0.5,
            protocol="different_protocol"
        )
    ]

    for experiment in experiments:

        episodic.add_experiment(experiment)

    failure.add_experiment(
        experiments[3],
        reference_loss=1.903852
    )

    failure.add_experiment(
        experiments[2]
    )

    failure.add_proposal_failure(
        reason="503 UNAVAILABLE",
        operational=True
    )

    retriever = MemoryRetriever(
        top_k_best=3,
        top_k_recent=3,
        top_k_findings=5,
        top_k_failures=3
    )

    # Test 1: Contextual relevance
    score = retriever.relevance_score(
        {
            "dropout": 0.0,
            "weight_decay": 0.01
        },
        {
            "learning_rate": 0.0007,
            "dropout": 0.0,
            "weight_decay": 0.01
        }
    )

    assert score == 1.0

    print("PASS: Contextual relevance")

    context = retriever.retrieve(
        episodic,
        semantic,
        failure,
        protocol_id
    )

    # Test 2: Best research state
    assert (
        context["research_state"]["best_experiment_id"]
        == 7
    )

    print("PASS: Best research state")

    # Test 3: Best experiment retrieval
    best_ids = [
        experiment["experiment_id"]
        for experiment in context["best_experiments"]
    ]

    assert best_ids == [7, 6, 3]

    print("PASS: Best experiment retrieval")

    # Test 4: Recent experiment retrieval
    recent_ids = [
        experiment["experiment_id"]
        for experiment in context["recent_experiments"]
    ]

    assert recent_ids == [0, 5]

    print("PASS: Recent experiment retrieval")

    # Test 5: No duplicate episodes
    assert not set(best_ids).intersection(recent_ids)

    print("PASS: Duplicate episodes excluded")

    # Test 6: Invalid and incompatible evidence
    all_ids = best_ids + recent_ids

    assert 4 not in all_ids
    assert 99 not in all_ids

    print("PASS: Invalid and incompatible evidence excluded")

    # Test 7: Semantic retrieval
    findings = context["semantic_findings"]

    assert 0 < len(findings) <= 5

    assert all(
        finding["protocol_id"] == protocol_id
        for finding in findings
    )

    assert all(
        4 not in finding["supporting_experiments"]
        for finding in findings
    )

    print("PASS: Semantic retrieval")

    # Test 8: Relevance ranking
    scores = [
        finding["relevance_score"]
        for finding in findings
    ]

    assert scores == sorted(scores, reverse=True)

    print("PASS: Relevance ranking")

    # Test 9: Scientific failure retrieval
    failures = context["scientific_failures"]

    assert len(failures) == 1
    assert failures[0]["experiment_id"] == 5

    print("PASS: Scientific failure retrieval")

    # Test 10: Empty memory
    empty_context = retriever.retrieve(
        EpisodicMemory(),
        SemanticMemory(),
        FailureMemory(),
        protocol_id
    )

    assert (
        empty_context["research_state"]["best_val_loss"]
        is None
    )

    assert empty_context["best_experiments"] == []

    print("PASS: Empty memory handling")

    print("\nRETRIEVED RESEARCH CONTEXT:")

    import json

    print(json.dumps(context, indent=4))

    print("\nALL MEMORY RETRIEVAL TESTS PASSED")


if __name__ == "__main__":

    test_episodic_memory()
    test_semantic_memory()
    test_failure_memory()
    test_memory_retrieval()
    