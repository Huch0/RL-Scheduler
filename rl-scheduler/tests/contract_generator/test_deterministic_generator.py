import pytest
from contract_generator import DeterministicGenerator
from config_path import INSTANCES_DIR

# Define the contracts file path (adjust as needed)
CONTRACTS_PATH = INSTANCES_DIR / "contracts" / "C-example0-5.json"


@pytest.fixture
def profit_functions():
    return DeterministicGenerator.load_profit_fn(CONTRACTS_PATH)


@pytest.fixture
def repetition():
    return DeterministicGenerator.load_repetition(CONTRACTS_PATH)


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
