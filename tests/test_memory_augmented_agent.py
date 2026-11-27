
import json
import tempfile

from unittest.mock import patch

from src.memory.manager import MemoryManager
from src.research_agent import ResearchAgent
from src.memory_augmented_agent import (
    MemoryAugmentedResearchAgent
)


def test_memory_augmented_agent():

    with tempfile.TemporaryDirectory() as directory:

        manager = MemoryManager(
            protocol_id="test_protocol_v1",
            memory_dir=directory
        )

        baseline = {
            "learning_rate": 0.001,
            "dropout": 0.1,
            "weight_decay": 0.01
        }

        manager.record_experiment({
            "experiment_id": 0,
            "status": "success",
            "parameters": baseline,
            "best_val_loss": 2.002828
        })

        manager.record_experiment({
            "experiment_id": 1,
            "status": "success",
            "parameters": {
                "learning_rate": 0.001,
                "dropout": 0.0,
                "weight_decay": 0.01
            },
            "best_val_loss": 1.903852
        })

        # Mock the parent constructor to avoid
        # creating a real Gemini API client.
        with patch.object(
            ResearchAgent,
            "__init__",
            return_value=None
        ):

            agent = MemoryAugmentedResearchAgent(
                memory_manager=manager
            )

        # Test 1: Prompt construction
        prompt = agent.build_prompt(
            history=[],
            baseline_config=baseline
        )

        assert "STRUCTURED EXPERIMENTAL MEMORY" in prompt

        print("PASS: Structured prompt construction")

        # Test 2: Research state injection
        assert "1.903852" in prompt

        print("PASS: Research state injection")

        # Test 3: Semantic knowledge injection
        assert '"semantic_findings"' in prompt
        assert '"supporting_experiments"' in prompt

        print("PASS: Semantic knowledge injection")

        # Test 4: Failure memory injection
        assert '"scientific_failures"' in prompt
        assert '"failure_summary"' in prompt

        print("PASS: Failure memory injection")

        # Test 5: Complete memory context
        context_json = json.dumps(
            manager.get_context(),
            indent=2
        )

        assert context_json in prompt

        print("PASS: Complete memory context injection")

        # Test 6: Simulate a new Gemini proposal
        mock_proposal = {
            "hypothesis": (
                "Reducing learning rate may "
                "improve convergence."
            ),
            "reasoning": (
                "Previous experiments suggest "
                "that zero dropout is promising."
            ),
            "parameters": {
                "learning_rate": 0.0008,
                "dropout": 0.0,
                "weight_decay": 0.01
            }
        }

        with patch.object(
            ResearchAgent,
            "propose",
            return_value=mock_proposal
        ):

            proposal = agent.propose_from_memory(
                baseline
            )

            assert proposal == mock_proposal

        print("PASS: New proposal accepted")

        # Test 7: Reject duplicate configuration
        duplicate_proposal = {
            **mock_proposal,
            "parameters": baseline
        }

        with patch.object(
            ResearchAgent,
            "propose",
            return_value=duplicate_proposal
        ):

            try:

                agent.propose_from_memory(baseline)

            except ValueError as error:

                assert "Duplicate" in str(error)

                print("PASS: Duplicate proposal rejected")

            else:

                raise AssertionError(
                    "Duplicate proposal was accepted"
                )

        print("\nALL MEMORY-AUGMENTED AGENT TESTS PASSED")


if __name__ == "__main__":

    test_memory_augmented_agent()
