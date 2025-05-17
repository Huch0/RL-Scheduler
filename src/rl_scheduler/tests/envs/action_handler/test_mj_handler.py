import pytest
from gymnasium import spaces

from rl_scheduler.scheduler.scheduler import Scheduler
from rl_scheduler.scheduler.slot_allocator import LinearSlotAllocator
from rl_scheduler.contract_generator import DeterministicGenerator
from rl_scheduler.config_path import INSTANCES_DIR
from rl_scheduler.envs.action_handler.mj_handler import MJHandler

# 테스트 전용 인스턴스 경로 (test_priority_rules_steps 와 동일)
CONTRACTS = INSTANCES_DIR / "contracts" / "C-test0.json"
MACHINES  = INSTANCES_DIR / "machines"  / "M-test0-2.json"
JOBS      = INSTANCES_DIR / "jobs"      / "J-test0-2.json"
OPS       = INSTANCES_DIR / "operations"/ "O-test0.json"

@pytest.fixture
def scheduler():
    gen = DeterministicGenerator(CONTRACTS)
    reps = gen.load_repetition()
    prof = gen.load_profit_fn()
    sched = Scheduler(
        machine_config_path=MACHINES,
        job_config_path=JOBS,
        operation_config_path=OPS,
        slot_allocator=LinearSlotAllocator,
    )
    sched.reset(reps, prof)
    return sched

def test_mj_handler_with_real_etd_priority_sequence(scheduler):
    handler = MJHandler(scheduler, priority_rule_id="etd")

    # 1) action space
    space = handler.create_action_space()
    assert isinstance(space, spaces.Discrete)
    assert space.n == handler.M * handler.J

    # STEP 0: 초기 우선순위
    # 각 job_template j에 대해 repetition r=0 이 선택되어야 함
    for action in range(space.n):
        m, j, r = handler.convert_action(action)
        assert r == 0

    # STEP 1: machine 0 에서 job0-inst0, job1-inst0 수행
    scheduler.step(chosen_machine_id=0, chosen_job_id=0, chosen_repetition=0)

    # 이제 job0-inst1 이 우선, job1-inst0 은 여전히 0
    for action in range(space.n):
        m, j, r = handler.convert_action(action)
        expected = 1 if j == 0 else 0
        assert r == expected

    # STEP 2: 두 번째 연산까지
    scheduler.step(chosen_machine_id=0, chosen_job_id=1, chosen_repetition=0)

    # 이제 최종 우선순위: job0-inst0, job1-inst1
    for action in range(space.n):
        m, j, r = handler.convert_action(action)
        expected = 0 if j == 0 else 1
        assert r == expected

def test_mj_handler_with_real_epv_priority_sequence(scheduler):
    handler = MJHandler(scheduler, priority_rule_id="epv")

    # action space 확인
    space = handler.create_action_space()
    assert isinstance(space, spaces.Discrete)
    assert space.n == handler.M * handler.J

    # STEP 0: 초기 EPV 우선순위 조회
    pri0 = handler.priority_rule.assign_priority()
    for action in range(space.n):
        _, j, r = handler.convert_action(action)
        assert r == pri0[j]

    # STEP 1: job0-inst0 의 첫 operation 수행
    scheduler.step(chosen_machine_id=0, chosen_job_id=0, chosen_repetition=0)

    pri1 = handler.priority_rule.assign_priority()
    for action in range(space.n):
        _, j, r = handler.convert_action(action)
        assert r == pri1[j]

    # STEP 2: job1-inst0 의 첫 operation 수행
    scheduler.step(chosen_machine_id=0, chosen_job_id=1, chosen_repetition=0)

    pri2 = handler.priority_rule.assign_priority()
    for action in range(space.n):
        _, j, r = handler.convert_action(action)
        assert r == pri2[j]
