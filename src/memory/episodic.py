
import json
import math
from copy import deepcopy
from pathlib import Path


class EpisodicMemory:
    """
    Store and retrieve individual experimental experiences.
    """

    def __init__(self, memory_path=None):

        self.memory_path = (
            Path(memory_path)
            if memory_path is not None
            else None
        )

        self.experiments = []

        if (
            self.memory_path is not None
            and self.memory_path.exists()
        ):
            self.load()

    def add_experiment(self, record):
        """
        Validate and store a new experimental record.
        """

        required_fields = [
            "experiment_id",
            "status",
            "parameters"
        ]

        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"Missing required field: {field}"
                )

        experiment_id = record["experiment_id"]

        if self.get_experiment(experiment_id) is not None:
            raise ValueError(
                f"Experiment {experiment_id} already exists"
            )

        if record["status"] not in ["success", "failed"]:
            raise ValueError("Invalid experiment status")

        if record["status"] == "success":

            loss = record.get("best_val_loss")

            if (
                not isinstance(loss, (int, float))
                or isinstance(loss, bool)
                or not math.isfinite(loss)
                or loss < 0
            ):
                raise ValueError("Invalid validation loss")

        episode = deepcopy(record)

        episode.setdefault("valid", True)
        episode.setdefault("invalid_reason", None)

        self.experiments.append(episode)

        self.save()

        return deepcopy(episode)

    def get_experiment(self, experiment_id):
        """
        Retrieve an experiment using its identifier.
        """

        for experiment in self.experiments:

            if experiment["experiment_id"] == experiment_id:
                return deepcopy(experiment)

        return None

    def get_all(self):
        """
        Retrieve the complete experimental history.
        """

        return deepcopy(self.experiments)

    def get_successful(self):
        """
        Retrieve successful and valid experiments.
        """

        return deepcopy([
            experiment
            for experiment in self.experiments
            if experiment["status"] == "success"
            and experiment.get("valid", True)
        ])

    def get_best(self, k=3):
        """
        Retrieve the k best valid experiments.
        """

        successful = self.get_successful()

        ranked = sorted(
            successful,
            key=lambda x: x["best_val_loss"]
        )

        return ranked[:k]

    def get_recent(self, k=3):
        """
        Retrieve the k most recent experiments.
        """

        if k <= 0:
            return []

        return deepcopy(self.experiments[-k:])

    def mark_invalid(self, experiment_id, reason):
        """
        Preserve an experiment but exclude it
        from valid scientific evidence.
        """

        for experiment in self.experiments:

            if experiment["experiment_id"] == experiment_id:

                experiment["valid"] = False
                experiment["invalid_reason"] = reason

                self.save()

                return

        raise ValueError(
            f"Experiment {experiment_id} not found"
        )

    def save(self):
        """
        Persist episodic memory to a JSON file.
        """

        if self.memory_path is None:
            return

        self.memory_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_path = self.memory_path.with_suffix(
            ".tmp"
        )

        temporary_path.write_text(
            json.dumps(
                self.experiments,
                indent=4,
                allow_nan=False
            ),
            encoding="utf-8"
        )

        temporary_path.replace(self.memory_path)

    def load(self):
        """
        Restore previously saved episodic memory.
        """

        with open(
            self.memory_path,
            encoding="utf-8"
        ) as file:

            self.experiments = json.load(file)

    def __len__(self):

        return len(self.experiments)
