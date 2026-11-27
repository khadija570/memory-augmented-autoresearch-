
from copy import deepcopy


class MemoryRetriever:
    """
    Retrieve relevant experimental knowledge
    for the next research iteration.
    """

    def __init__(
        self,
        top_k_best=3,
        top_k_recent=3,
        top_k_findings=5,
        top_k_failures=3
    ):

        self.top_k_best = top_k_best
        self.top_k_recent = top_k_recent
        self.top_k_findings = top_k_findings
        self.top_k_failures = top_k_failures

    def relevance_score(
        self,
        fixed_parameters,
        current_parameters
    ):
        """
        Measure contextual similarity between
        experimental conditions.
        """

        if not fixed_parameters:
            return 0.0

        matches = sum(
            1
            for key, value in fixed_parameters.items()
            if current_parameters.get(key) == value
        )

        return matches / len(fixed_parameters)

    def retrieve_findings(
        self,
        semantic_memory,
        current_parameters,
        protocol_id
    ):
        """
        Retrieve relevant semantic findings.
        """

        candidates = []

        for finding in semantic_memory.get_all():

            # Exclude incompatible experimental protocols
            if finding.get("protocol_id") != protocol_id:
                continue

            score = self.relevance_score(
                finding["fixed_parameters"],
                current_parameters
            )

            candidate = deepcopy(finding)

            candidate["relevance_score"] = score

            candidates.append(candidate)

        # Prioritize context, then observed effect
        ranked = sorted(
            candidates,
            key=lambda item: (
                item["relevance_score"],
                abs(item["improvement_pct"])
            ),
            reverse=True
        )

        return ranked[:self.top_k_findings]

    def retrieve_failures(
        self,
        failure_memory,
        current_parameters,
        protocol_id,
        valid_experiment_ids
    ):
        """
        Retrieve relevant scientific failures.
        """

        candidates = []

        for failure in (
            failure_memory.get_scientific_failures()
        ):

            if failure.get("protocol_id") != protocol_id:
                continue

            if (
                failure.get("experiment_id")
                not in valid_experiment_ids
            ):
                continue

            score = self.relevance_score(
                failure.get("parameters", {}),
                current_parameters
            )

            candidate = deepcopy(failure)

            candidate["relevance_score"] = score

            candidates.append(candidate)

        # Prefer recent failures when relevance is equal
        ranked = sorted(
            reversed(candidates),
            key=lambda item: item["relevance_score"],
            reverse=True
        )

        return ranked[:self.top_k_failures]

    def compact_experiment(self, experiment):
        """
        Keep only the information needed
        for research reasoning.
        """

        fields = (
            "experiment_id",
            "parameters",
            "best_val_loss",
            "best_step",
            "decision",
            "protocol_id"
        )

        return {
            key: deepcopy(experiment[key])
            for key in fields
            if key in experiment
        }

    def retrieve(
        self,
        episodic_memory,
        semantic_memory,
        failure_memory,
        protocol_id
    ):
        """
        Construct the structured research context.
        """

        if not protocol_id:
            raise ValueError("A protocol ID is required")

        valid_experiments = [
            experiment
            for experiment in (
                episodic_memory.get_successful()
            )
            if experiment.get("protocol_id") == protocol_id
        ]

        ranked = sorted(
            valid_experiments,
            key=lambda item: item["best_val_loss"]
        )

        best = ranked[0] if ranked else None

        current_parameters = (
            best["parameters"]
            if best is not None
            else {}
        )

        best_experiments = ranked[:self.top_k_best]

        # Avoid repeating the best experiments
        # in the recent-experiment section
        best_ids = {
            experiment["experiment_id"]
            for experiment in best_experiments
        }

        recent_candidates = [
            experiment
            for experiment in valid_experiments
            if experiment["experiment_id"] not in best_ids
        ]

        recent_experiments = (
            recent_candidates[-self.top_k_recent:]
            if self.top_k_recent > 0
            else []
        )

        valid_ids = {
            experiment["experiment_id"]
            for experiment in valid_experiments
        }

        # Refresh findings to exclude stale evidence
        semantic_memory.extract_findings(
            valid_experiments
        )

        findings = self.retrieve_findings(
            semantic_memory,
            current_parameters,
            protocol_id
        )

        failures = self.retrieve_failures(
            failure_memory,
            current_parameters,
            protocol_id,
            valid_ids
        )

        context = {

            "research_state": {
                "protocol_id": protocol_id,
                "total_valid_experiments": len(
                    valid_experiments
                ),
                "best_experiment_id": (
                    best["experiment_id"]
                    if best is not None
                    else None
                ),
                "best_val_loss": (
                    best["best_val_loss"]
                    if best is not None
                    else None
                ),
                "best_parameters": deepcopy(
                    current_parameters
                )
            },

            "best_experiments": [
                self.compact_experiment(experiment)
                for experiment in best_experiments
            ],

            "recent_experiments": [
                self.compact_experiment(experiment)
                for experiment in recent_experiments
            ],

            "semantic_findings": findings,

            "scientific_failures": failures,

            "failure_summary": (
                failure_memory.get_summary()
            )
        }

        return context
