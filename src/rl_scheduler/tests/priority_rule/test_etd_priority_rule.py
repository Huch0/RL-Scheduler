import pytest
from rl_scheduler.priority_rule.job_type_scope.etd_priority_rule import ETDPriorityRule
from rl_scheduler.scheduler.scheduler import Scheduler
from rl_scheduler.scheduler.slot_allocator import LinearSlotAllocator
from rl_scheduler.contract_generator import DeterministicGenerator as ContractGenerator
from rl_scheduler.config_path import INSTANCES_DIR

# 경로 정의
CONTRACTS_PATH = INSTANCES_DIR / "contracts" / "C-example0-5.json"


@pytest.fixture
def contracts():
    repetitions = ContractGenerator.load_repetition(CONTRACTS_PATH)
    profit_functions = ContractGenerator.load_profit_fn(CONTRACTS_PATH)
    return repetitions, profit_functions


@pytest.fixture
def scheduler(contracts):
    reps, profits = contracts
    sched = Scheduler(
        machine_config_path=INSTANCES_DIR / "machines" / "M-example0-3.json",
        job_config_path=INSTANCES_DIR / "jobs" / "J-example0-5.json",
        operation_config_path=INSTANCES_DIR / "operations" / "O-example0.json",
        slot_allocator=LinearSlotAllocator,
    )
    sched.reset(reps, profits)
    return sched


def test_assign_priority_initial(scheduler):
    """
    초기 상태에서는 모든 그룹에서 첫 번째 인스턴스가 선택되어야 함
    """
    rule = ETDPriorityRule(scheduler)
    assert rule.assign_priority() == [0, 0, 0, 0, 0]


def test_assign_priority_after_some_completed(scheduler):
    """
    일부 인스턴스를 완료 처리한 후 우선순위 확인
    """
    # job0과 job2의 첫 번째 인스턴스를 완료 표시
    scheduler.job_instances[0][0].completed = True
    scheduler.job_instances[2][0].completed = True
    rule = ETDPriorityRule(scheduler)
    priorities = rule.assign_priority()
    # job0과 job2는 두 번째 인스턴스(인덱스 1)가 선택되어야 함
    assert priorities[0] == 1
    assert priorities[2] == 1
    # 나머지 그룹은 첫 번째(0)
    assert priorities == [1, 0, 1, 0, 0]


def test_assign_priority_all_completed(scheduler):
    """
    모든 인스턴스를 완료 처리하면 -1 반환
    """
    for group in scheduler.job_instances:
        for job in group:
            job.completed = True
    rule = ETDPriorityRule(scheduler)
    assert rule.assign_priority() == [-1, -1, -1, -1, -1]
