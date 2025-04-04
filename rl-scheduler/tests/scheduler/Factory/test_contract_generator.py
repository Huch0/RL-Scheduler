import pytest
from pathlib import Path
from scheduler.Factory.ContractGenerator import ContractGenerator

# Define the contracts file path (adjust as needed)
CONTRACTS_PATH = Path(__file__).parent.parent.parent.parent / "instances" / "Contracts" / "C-example0-5.json"

@pytest.fixture
def profit_functions():
    return ContractGenerator.load_profit_fn(CONTRACTS_PATH)

@pytest.fixture
def repeatitions():
    return ContractGenerator.load_repeatition(CONTRACTS_PATH)

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

def test_load_repeatition(repeatitions):
    # Each job group has 3 contracts -> repeatitions should be [3, 3, 3, 3, 3]
    assert repeatitions == [3, 3, 3, 3, 3]
