
from itertools import combinations
from copy import deepcopy
import math


class SemanticMemory:
    """
    Extract contextual, evidence-backed findings
    from valid experimental experiences.
    """

    PARAMETERS = (
        "learning_rate",
        "dropout",
        "weight_decay"
    )

    def __init__(self, min_effect_pct=0.1):

        self.findings = []

        # Threshold for practically small effects
        self.min_effect_pct = min_effect_pct

    def compare_experiments(self, exp_a, exp_b):
        """
        Compare experiments that differ in exactly
        one hyperparameter.
        """

        # Exclude unsuccessful or invalid experiments
        for experiment in (exp_a, exp_b):

            if experiment["status"] != "success":
                return None

            if not experiment.get("valid", True):
                return None

        # Require a matching experimental protocol
        protocol_a = exp_a.get("protocol_id")
        protocol_b = exp_b.get("protocol_id")

        if protocol_a is None or protocol_a != protocol_b:
            return None

        params_a = exp_a["parameters"]
        params_b = exp_b["parameters"]

        # Identify changed hyperparameters
        changed = [
            parameter
            for parameter in self.PARAMETERS
            if params_a[parameter] != params_b[parameter]
        ]

        # Isolate one hyperparameter
        if len(changed) != 1:
            return None

        parameter = changed[0]

        # Orient comparisons from lower to higher value
        if params_a[parameter] > params_b[parameter]:

            exp_a, exp_b = exp_b, exp_a
            params_a, params_b = params_b, params_a

        loss_a = exp_a["best_val_loss"]
        loss_b = exp_b["best_val_loss"]

        if (
            not math.isfinite(loss_a)
            or not math.isfinite(loss_b)
            or loss_a <= 0
            or loss_b < 0
        ):
            return None

        improvement_pct = (
            (loss_a - loss_b) / loss_a
        ) * 100

        # Classify the observed effect
        if abs(improvement_pct) < self.min_effect_pct:

            effect = "negligible"

        elif improvement_pct > 0:

            effect = "improved"

        else:

            effect = "degraded"

        # Preserve the experimental context
        fixed_parameters = {
            key: value
            for key, value in params_a.items()
            if key != parameter
        }

        finding = {
            "parameter": parameter,

            "from_value": params_a[parameter],
            "to_value": params_b[parameter],

            "fixed_parameters": fixed_parameters,

            "loss_before": loss_a,
            "loss_after": loss_b,

            "improvement_pct": improvement_pct,

            "effect": effect,

            "supporting_experiments": [
                exp_a["experiment_id"],
                exp_b["experiment_id"]
            ],

            "protocol_id": protocol_a,

            "evidence_strength": "preliminary"
        }

        finding["observation"] = (
            f"Increasing {parameter} from "
            f"{params_a[parameter]} to "
            f"{params_b[parameter]} resulted in "
            f"a {effect} effect on validation loss "
            f"under the tested conditions."
        )

        return finding

    def extract_findings(self, experiments):
        """
        Extract findings from all valid pairs
        of experimental experiences.
        """

        self.findings = []

        for exp_a, exp_b in combinations(experiments, 2):

            finding = self.compare_experiments(
                exp_a,
                exp_b
            )

            if finding is not None:

                self.findings.append(finding)

        return self.get_all()

    def get_all(self):
        """
        Return all extracted findings.
        """

        return deepcopy(self.findings)

    def get_by_parameter(self, parameter):
        """
        Retrieve knowledge about one hyperparameter.
        """

        return deepcopy([
            finding
            for finding in self.findings
            if finding["parameter"] == parameter
        ])

    def get_strongest(self, k=5):
        """
        Retrieve findings with the largest
        observed absolute effect.
        """

        ranked = sorted(
            self.findings,
            key=lambda finding: abs(
                finding["improvement_pct"]
            ),
            reverse=True
        )

        return deepcopy(ranked[:max(0, k)])

    def __len__(self):

        return len(self.findings)
