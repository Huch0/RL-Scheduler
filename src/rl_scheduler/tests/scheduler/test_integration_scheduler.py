import pytest
from rl_scheduler.scheduler.scheduler import Scheduler
from rl_scheduler.scheduler.slot_allocator import LinearSlotAllocator
from rl_scheduler.contract_generator import DeterministicGenerator
from rl_scheduler.config_path import INSTANCES_DIR

# 통합 테스트: json 파일들로부터 템플릿을 파싱, reset으로 instance 할당, 그리고 step 메서드 검증.


@pytest.fixture
def scheduler_integration():
    machine_config = INSTANCES_DIR / "machines" / "M-example0-3.json"
    job_config = INSTANCES_DIR / "jobs" / "J-example0-5.json"
    operation_config = INSTANCES_DIR / "operations" / "O-example0.json"
    # Scheduler 초기화 시 템플릿이 파싱됨.
    sched = Scheduler(
        machine_config_path=machine_config,
        job_config_path=job_config,
        operation_config_path=operation_config,
        slot_allocator=LinearSlotAllocator,
    )
    return sched


@pytest.fixture
def contracts():
    contract_file = INSTANCES_DIR / "contracts" / "C-example0-5.json"
    deterministicGenerator = DeterministicGenerator(contract_file)
    repetitions = deterministicGenerator.load_repetition()
    profit_functions = deterministicGenerator.load_profit_fn()
    return repetitions, profit_functions


def test_scheduler_integration_init_and_reset(scheduler_integration, contracts):
    repetitions, profit_functions = contracts
    # reset 전에는 instance 변수들이 None이어야 함.
    assert scheduler_integration.machine_instances is None
    assert scheduler_integration.job_instances is None

    # reset 메서드를 통해 instance 할당
    scheduler_integration.reset(repetitions, profit_functions)
    assert scheduler_integration.machine_instances is not None
    assert scheduler_integration.job_instances is not None

    # 할당 결과 디버깅을 위해 print (pytest -s 옵션으로 출력 확인 가능)
    print("=== machine Instances ===")
    for m in scheduler_integration.machine_instances:
        print(m)
    print("=== job Instances ===")
    for job_group in scheduler_integration.job_instances:
        for job in job_group:
            print(job)


def test_scheduler_integration_step(scheduler_integration, contracts):
    repetitions, profit_functions = contracts
    scheduler_integration.reset(repetitions, profit_functions)

    # 세 액션: (0, 0, 0), (0, 0, 0), (1, 0, 0)
    actions = [(0, 0, 0), (0, 0, 0), (1, 0, 0)]
    for chosen_machine_id, chosen_job_id, chosen_repetition in actions:
        chosen_op = scheduler_integration.find_op_instance_by_action(
            +chosen_job_id, chosen_repetition
        )

        scheduler_integration.step(
            chosen_machine_id=chosen_machine_id,
            chosen_job_id=chosen_job_id,
            chosen_repetition=chosen_repetition,
        )
        assert chosen_op.start_time >= 0
        assert chosen_op.end_time > chosen_op.start_time
        assert chosen_op.duration == (chosen_op.end_time - chosen_op.start_time)
        assert (
            chosen_op.processing_machine.machine_template.machine_template_id
            == chosen_machine_id
        )
        assert (
            chosen_op.job_instance
            == scheduler_integration.job_instances[chosen_job_id][chosen_repetition]
        )
        next_op = chosen_op.successor
        if next_op:
            assert next_op.earliest_start_time == chosen_op.end_time
            assert next_op.predecessor == chosen_op
        else:
            assert chosen_op.successor is None
            # 마지막 작업의 successor는 None이어야 함.

    # 각 머신의 슬롯 할당 상태 확인
    print("=== Slot Allocation Status ===")
    for machine in scheduler_integration.machine_instances:
        print(f"machine {machine.machine_template.machine_template_id}:")
        for op in machine.assigned_operations:
            print(
                f"""  Operation
                {op.operation_template.operation_template_id}: Start Time
                {op.start_time}, End Time {op.end_time}"""
            )
    print("===")
