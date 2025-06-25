import pytest
import math
from rl_scheduler.priority_rule.job_type_scope.edd_priority_rule import EDDPriorityRule
from rl_scheduler.priority_rule.job_type_scope.etd_priority_rule import ETDPriorityRule
from rl_scheduler.priority_rule.job_type_scope.epv_priority_rule import EPVPriorityRule
from rl_scheduler.scheduler.scheduler import Scheduler
from rl_scheduler.scheduler.slot_allocator import LinearSlotAllocator
from rl_scheduler.contract_generator import DeterministicGenerator
from rl_scheduler.config_path import INSTANCES_DIR

# 테스트 전용 인스턴스 경로
CONTRACTS = INSTANCES_DIR / "contracts" / "C-test0.json"
MACHINES  = INSTANCES_DIR / "machines"  / "M-test0-2.json"
JOBS      = INSTANCES_DIR / "jobs"      / "J-test0-2.json"
OPS       = INSTANCES_DIR / "operations"/ "O-test0.json"

@pytest.fixture
def scheduler():
    deterministicGenerator = DeterministicGenerator(CONTRACTS)
    reps = deterministicGenerator.load_repetition()
    prof = deterministicGenerator.load_profit_fn()
    sched = Scheduler(
        machine_config_path=MACHINES,
        job_config_path=JOBS,
        operation_config_path=OPS,
        slot_allocator=LinearSlotAllocator,
    )
    sched.reset(reps, prof)
    return sched

def test_priority_rules_over_steps(scheduler):
    # --- STEP 0: 초기 상태 ---
    edd0 = EDDPriorityRule(scheduler)
    etd0 = ETDPriorityRule(scheduler)
    epv0 = EPVPriorityRule(scheduler)

    # 1) assign_priority
    assert edd0.assign_priority() == [0, 0]
    assert etd0.assign_priority() == [0, 0]
    assert epv0.assign_priority() == [0, 0]

    # 2) compute_metrics (값 자체도 체크)
    #   EDD: deadline 그대로
    m_edd0 = edd0.compute_metrics()
    assert m_edd0[(0,0)] == 15
    assert m_edd0[(0,1)] == 16
    assert m_edd0[(1,0)] == 25
    assert m_edd0[(1,1)] == 26

    #   ETD: 예제값과 비교 (step0)
    m_etd0 = etd0.compute_metrics()
    # (3/15)*(3-15) = -2.4, (7/20)*(7-25) = -6.3
    assert m_etd0[(0,0)] == pytest.approx(-2.4, rel=1e-3)
    assert m_etd0[(1,0)] == pytest.approx(-6.3, rel=1e-3)

    #   EPV: 아직 lateness 없음 → price 그대로
    m_epv0 = epv0.compute_metrics()
    assert m_epv0[(0,0)] == pytest.approx(1000.0)
    assert m_epv0[(1,0)] == pytest.approx(1500.0)

    # --- STEP 1: 각 그룹 첫 번째 operation 완료 표시 ---
    # machine 0에서 job0-inst0, job1-inst0의 첫 operation 수행
    scheduler.step(chosen_machine_id=0, chosen_job_id=0, chosen_repetition=0)

    edd1 = EDDPriorityRule(scheduler)
    etd1 = ETDPriorityRule(scheduler)
    epv1 = EPVPriorityRule(scheduler)

    assert edd1.assign_priority() == [0, 0]
    
    # ETD step1 예제
    m_etd1 = etd1.compute_metrics()

    assert etd1.assign_priority() == [1, 0]
    # J_11의 ETD =  (8/15)*(8-15)≈-3.733
    assert m_etd1[(0,0)] == pytest.approx(-3.733, rel=1e-3)
    # J_12의 ETD = (3/15)*((3+6)/2-16)≈-2.3
    assert m_etd1[(0,1)] == pytest.approx(-2.3, rel=1e-3)
    # J_21의 ETD = (7/20)*(10-25)≈-5.25
    assert m_etd1[(1,0)] == pytest.approx(-5.25,  rel=1e-3)
    # J_22의 ETD = (7/20)*(10-26)≈-5.6
    assert m_etd1[(1,1)] == pytest.approx(-5.6,  rel=1e-3)

    # EPV step1 예제
    m_epv1 = epv1.compute_metrics()
    assert epv1.assign_priority() == [1, 0]
    assert m_epv1[(0,0)] == pytest.approx(1000.0)
    assert m_epv1[(0,1)] == pytest.approx(975.0)
    assert m_epv1[(1,0)] == pytest.approx(1500.0)
    assert m_epv1[(1,1)] == pytest.approx(1500.0)



    # --- STEP 2: 각 그룹 두 번째 operation 까지 완료 표시 ---
    # machine 0에서 job0-inst0, job1-inst0의 두 번째 operation 수행
    scheduler.step(chosen_machine_id=0, chosen_job_id=1, chosen_repetition=0)

    edd2 = EDDPriorityRule(scheduler)
    etd2 = ETDPriorityRule(scheduler)
    epv2 = EPVPriorityRule(scheduler)

    # EDD, ETD 우선순위 여전히 첫 인스턴스
    assert edd2.assign_priority() == [0, 0]

    assert etd2.assign_priority() == [0, 1]
    # ETD step2 예제
    m_etd1 = etd1.compute_metrics()
    # J_11의 ETD = (8/15)*(15-15)≈0
    assert m_etd1[(0,0)] == pytest.approx(0, rel=1e-3)
    # J_12의 ETD = (3/15)*((13+3)/2-16)≈-1.6
    assert m_etd1[(0,1)] == pytest.approx(-1.6, rel=1e-3)
    # J_21의 ETD = (17/20)*(20-25)≈-4.25
    assert m_etd1[(1,0)] == pytest.approx(-4.25,  rel=1e-3)
    # J_22의 ETD = (7/20)*(17-26)≈-3.15
    assert m_etd1[(1,1)] == pytest.approx(-3.15,  rel=1e-3)


    # EPV step2 예제
    pri_epv2 = epv2.assign_priority()
    assert pri_epv2 == [0, 1]

    m_epv2 = epv2.compute_metrics()
    assert m_epv2[(0,0)] == pytest.approx(650.0)
    assert m_epv2[(0,1)] == pytest.approx(800.0)
    assert m_epv2[(1,0)] == pytest.approx(1500.0)
    assert m_epv2[(1,1)] == pytest.approx(1200.0)