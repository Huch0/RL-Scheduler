import pytest
from rl_scheduler.contract_generator import DeterministicGenerator
from rl_scheduler.config_path import INSTANCES_DIR

# Define the contracts file path (adjust as needed)
CONTRACTS_PATH = INSTANCES_DIR / "contracts" / "C-example0-5.json"


@pytest.fixture
def profit_functions():
    deterministic_generator = DeterministicGenerator(CONTRACTS_PATH)
    return deterministic_generator.load_profit_fn()


@pytest.fixture
def repetition():
    deterministic_generator = DeterministicGenerator(CONTRACTS_PATH)
    return deterministic_generator.load_repetition()


def test_load_profit_fn(profit_functions):
    # Check that there are 5 job groups (jobs job_0 to job_4)
    assert len(profit_functions) == 5
    # Each job group has 3 profit functions
    for group in profit_functions:
        assert len(group) == 3
    # Verify properties of the first profit function in the first group (job_0)
    pf = profit_functions[0][0]
    assert pf.job_instance_id == 0
    assert pf.price == 1000.0
    assert pf.deadline == 15
    assert pf.late_penalty == 50.0


def test_load_repetition(repetition):
    # Each job group has 3 contracts -> repetition should be [3, 3, 3, 3, 3]
    assert repetition == [3, 3, 3, 3, 3]
