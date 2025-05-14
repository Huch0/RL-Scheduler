# rl_scheduler/priority_rule/job_type_scope/utils.py
from statistics import mean
from typing import List
from rl_scheduler.scheduler.slot_allocator.linear_slot_allocator import LinearSlotAllocator


def earliest_finish_on_machine(machine, op) -> int:
    """
    op 가 machine 위에서 종료될 수 있는 가장 빠른 시각 계산.
    """
    slot, _ = LinearSlotAllocator.find_slot(machine, op)
    return slot["end_time"]


def mean_first_op_finish(op, machines) -> float:
    """지원 가능한 모든 머신에서 op 종료 시각의 평균."""
    finishes = [
        earliest_finish_on_machine(m, op)
        for m in machines
        if op.type_code in m.machine_template.supported_operation_type_codes
    ]
    return mean(finishes) if finishes else float("inf")


def estimated_job_finish(job, machines) -> float:
    """
    ① 첫 미완료 op 평균 종료시각 + ② 나머지 op duration 합.
    """
    remaining = [op for op in job.operation_instance_sequence if op.end_time is None]
    if not remaining:
        return job.end_time or float("inf")

    first_finish = mean_first_op_finish(remaining[0], machines)
    rest_dur = sum(op.duration for op in remaining[1:])
    return first_finish + rest_dur
