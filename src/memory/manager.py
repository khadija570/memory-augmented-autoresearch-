
from copy import deepcopy
from pathlib import Path

from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.memory.failure import FailureMemory
from src.memory.retrieval import MemoryRetriever


class MemoryManager:
    """
    Coordinate experimental memory components
    throughout the autonomous research process.
    """

    def __init__(
        self,
        protocol_id,
        memory_dir="results/memory"
    ):

        if not isinstance(protocol_id, str) or not protocol_id.strip():
            raise ValueError("A valid protocol ID is required")

        self.protocol_id = protocol_id

        self.memory_dir = Path(memory_dir)

        self.memory_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.episodic = EpisodicMemory(
            self.memory_dir / "episodic_memory.json"
        )

        self.semantic = SemanticMemory()

        self.failure = FailureMemory()

        self.retriever = MemoryRetriever()

        # Restore derived knowledge from saved episodes
        self.rebuild_memory()

    def get_valid_experiments(self):
        """
        Retrieve valid experiments belonging
        to the current experimental protocol.
        """

        return [
            experiment
            for experiment in self.episodic.get_successful()
            if experiment.get("protocol_id")
            == self.protocol_id
        ]

    def get_best_experiment(self):
        """
        Retrieve the current best valid experiment.
        """

        experiments = self.get_valid_experiments()

        if not experiments:
            return None

        return min(
            experiments,
            key=lambda item: item["best_val_loss"]
        )

    def get_best_loss(self):
        """
        Retrieve the current best validation loss.
        """

        best = self.get_best_experiment()

        if best is None:
            return None

        return best["best_val_loss"]

    def has_tested(self, parameters):
        """
        Check the complete valid experimental history
        for an identical configuration.
        """

        for experiment in self.get_valid_experiments():

            if experiment["parameters"] == parameters:
                return True

        return False

    def record_experiment(self, record):
        """
        Record an experiment and update
        all derived memory components.
        """

        episode = deepcopy(record)

        supplied_protocol = episode.get(
            "protocol_id",
            self.protocol_id
        )

        if supplied_protocol != self.protocol_id:
            raise ValueError("Incompatible experimental protocol")

        episode["protocol_id"] = self.protocol_id

        episode.setdefault("valid", True)

        episode.setdefault("invalid_reason", None)

        # Store the incumbent before the new experiment
        best = self.get_best_experiment()

        reference_loss = (
            best["best_val_loss"]
            if best is not None
            else None
        )

        episode["reference_loss"] = reference_loss

        episode["reference_experiment_id"] = (
            best["experiment_id"]
            if best is not None
            else None
        )

        episode["improvement_pct"] = None

        # Determine the research decision
        if not episode["valid"]:

            episode["decision"] = "INVALID"

        elif episode["status"] == "failed":

            episode["decision"] = "FAILED"

        elif episode["status"] == "success":

            new_loss = episode["best_val_loss"]

            if reference_loss is None:

                episode["decision"] = "KEEP"

            else:

                if reference_loss > 0:

                    episode["improvement_pct"] = (
                        (reference_loss - new_loss)
                        / reference_loss
                    ) * 100

                episode["decision"] = (
                    "KEEP"
                    if new_loss < reference_loss
                    else "DISCARD"
                )

        else:

            raise ValueError("Unknown experiment status")

        # Persist the experimental experience
        stored = self.episodic.add_experiment(
            episode
        )

        # Reconstruct derived scientific knowledge
        self.rebuild_memory()

        return stored

    def record_proposal_failure(
        self,
        reason,
        parameters=None,
        attempt=None,
        operational=False
    ):
        """
        Record a failure before experiment execution.
        """

        return self.failure.add_proposal_failure(
            reason=reason,
            parameters=parameters,
            attempt=attempt,
            operational=operational
        )

    def rebuild_memory(self):
        """
        Reconstruct semantic and experimental failure
        memory from the saved episodic history.
        """

        # Preserve proposal/API failures during this session
        proposal_failures = [
            item
            for item in self.failure.get_all()
            if item.get("experiment_id") is None
        ]

        self.failure = FailureMemory()

        self.failure.failures.extend(
            proposal_failures
        )

        experiments = [
            experiment
            for experiment in self.episodic.get_all()
            if experiment.get("protocol_id")
            == self.protocol_id
        ]

        incumbent_loss = None

        for experiment in experiments:

            # Recalculate the historical reference
            # using only valid preceding experiments
            reference = incumbent_loss

            self.failure.add_experiment(
                experiment,
                reference_loss=(
                    reference
                    if reference is not None
                    and reference > 0
                    else None
                )
            )

            if (
                experiment["status"] == "success"
                and experiment.get("valid", True)
            ):

                loss = experiment["best_val_loss"]

                if (
                    incumbent_loss is None
                    or loss < incumbent_loss
                ):

                    incumbent_loss = loss

        self.semantic.extract_findings(
            self.get_valid_experiments()
        )

    def invalidate_experiment(
        self,
        experiment_id,
        reason
    ):
        """
        Exclude unreliable experimental evidence
        and rebuild the derived knowledge.
        """

        experiment = self.episodic.get_experiment(
            experiment_id
        )

        if experiment is None:
            raise ValueError("Experiment not found")

        if experiment.get("protocol_id") != self.protocol_id:
            raise ValueError("Incompatible experimental protocol")

        self.episodic.mark_invalid(
            experiment_id,
            reason
        )

        self.rebuild_memory()

    def get_context(self):
        """
        Generate the structured research context
        for the next LLM proposal.
        """

        return self.retriever.retrieve(
            self.episodic,
            self.semantic,
            self.failure,
            self.protocol_id
        )

    def get_summary(self):
        """
        Summarize the current memory state.
        """

        best = self.get_best_experiment()

        experiments = [
            experiment
            for experiment in self.episodic.get_all()
            if experiment.get("protocol_id")
            == self.protocol_id
        ]

        return {
            "protocol_id": self.protocol_id,

            "total_experiments": len(experiments),

            "valid_experiments": len(
                self.get_valid_experiments()
            ),

            "semantic_findings": len(self.semantic),

            "total_failures": len(self.failure),

            "failure_summary": (
                self.failure.get_summary()
            ),

            "best_experiment": best
        }
