
from src.experiment_runner import (
    validate_proposal,
    run_experiment,
)

def test_validation():
    valid = {
        "learning_rate": 0.0007,
        "dropout": 0.1,
        "weight_decay": 0.01,
    }

    assert validate_proposal(valid) == valid

    try:
        validate_proposal({
            "learning_rate": 10.0,
            "dropout": 0.1,
            "weight_decay": 0.01,
        })
    except ValueError:
        print("Invalid proposal correctly rejected")
    else:
        raise AssertionError("Invalid proposal was accepted")

if __name__ == "__main__":
    test_validation()
    print("Experiment runner validation test passed!")