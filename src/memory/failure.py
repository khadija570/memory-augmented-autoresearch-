
from copy import deepcopy
import math


class FailureMemory:
    """
    Classify and retrieve unsuccessful research experiences
    without confusing scientific and operational failures.
    """

    def __init__(self, tolerance_pct=0.1):

        self.failures = []
        self.tolerance_pct = tolerance_pct

    def classify_experiment(
        self,
        experiment,
        reference_loss=None
    ):
        """
        Classify an experimental outcome.
        """

        failure = {
            "experiment_id": experiment["experiment_id"],
            "parameters": deepcopy(
                experiment.get("parameters", {})
            ),
            "protocol_id": experiment.get("protocol_id"),
            "hypothesis": experiment.get("hypothesis"),
            "reference_loss": reference_loss,
            "observed_loss": experiment.get("best_val_loss")
        }

        # Case 1: Invalid scientific evidence
        if not experiment.get("valid", True):

            failure.update({
                "failure_type": "invalid_experiment",
                "reason": experiment.get(
                    "invalid_reason"
                ) or "Invalid experimental evidence",
                "scientific_evidence": False,
                "recommended_action": "Fix and revalidate"
            })

            return failure

        # Case 2: Execution failure
        if experiment["status"] == "failed":

            failure.update({
                "failure_type": "operational_failure",
                "reason": experiment.get(
                    "error",
                    "Unknown execution error"
                ),
                "scientific_evidence": False,
                "recommended_action": (
                    "Diagnose and retry if appropriate"
                )
            })

            return failure

        if experiment["status"] != "success":
            raise ValueError("Unknown experiment status")

        # No reference means no performance comparison
        if reference_loss is None:
            return None

        observed_loss = experiment["best_val_loss"]

        if (
            not math.isfinite(reference_loss)
            or reference_loss <= 0
            or not math.isfinite(observed_loss)
            or observed_loss < 0
        ):
            raise ValueError("Invalid loss value")

        improvement_pct = (
            (reference_loss - observed_loss)
            / reference_loss
        ) * 100

        failure["improvement_pct"] = improvement_pct

        # Any strict improvement is not a failure
        if improvement_pct > 0:
            return None

        # Case 3: Meaningful degradation
        if improvement_pct < -self.tolerance_pct:

            failure.update({
                "failure_type": "scientific_failure",
                "reason": (
                    "Validation loss increased "
                    "relative to the reference"
                ),
                "scientific_evidence": True,
                "recommended_action": (
                    "Use the observation to guide exploration"
                )
            })

        # Case 4: No improvement or small degradation
        else:

            failure.update({
                "failure_type": "no_improvement",
                "reason": (
                    "No improvement over the reference"
                ),
                "scientific_evidence": True,
                "recommended_action": (
                    "Explore another configuration"
                )
            })

        return failure

    def add_experiment(
        self,
        experiment,
        reference_loss=None
    ):
        """
        Classify and store an experimental failure.
        """

        failure = self.classify_experiment(
            experiment,
            reference_loss
        )

        if failure is not None:

            self.failures.append(
                deepcopy(failure)
            )

        return deepcopy(failure)

    def add_proposal_failure(
        self,
        reason,
        parameters=None,
        attempt=None,
        operational=False
    ):
        """
        Record a failure before experiment execution.
        """

        failure = {
            "experiment_id": None,
            "attempt": attempt,
            "parameters": deepcopy(parameters),
            "failure_type": (
                "operational_failure"
                if operational
                else "proposal_failure"
            ),
            "reason": str(reason),
            "scientific_evidence": False,
            "recommended_action": (
                "Retry with bounded backoff"
                if operational
                else "Generate a valid new proposal"
            )
        }

        self.failures.append(failure)

        return deepcopy(failure)

    def get_all(self):
        """
        Retrieve all recorded failures.
        """

        return deepcopy(self.failures)

    def get_by_type(self, failure_type):
        """
        Retrieve failures belonging to one category.
        """

        return deepcopy([
            failure
            for failure in self.failures
            if failure["failure_type"] == failure_type
        ])

    def get_scientific_failures(self):
        """
        Retrieve failures that contain valid
        scientific observations.
        """

        return deepcopy([
            failure
            for failure in self.failures
            if failure["scientific_evidence"]
        ])

    def get_recent(self, k=3):
        """
        Retrieve the most recent failures.
        """

        if k <= 0:
            return []

        return deepcopy(self.failures[-k:])

    def get_summary(self):
        """
        Count failures by category.
        """

        summary = {
            "scientific_failure": 0,
            "no_improvement": 0,
            "operational_failure": 0,
            "invalid_experiment": 0,
            "proposal_failure": 0
        }

        for failure in self.failures:

            category = failure["failure_type"]

            summary[category] += 1

        return summary

    def __len__(self):

        return len(self.failures)
