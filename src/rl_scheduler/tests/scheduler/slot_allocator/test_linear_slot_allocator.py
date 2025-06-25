import pytest
from unittest.mock import MagicMock
from rl_scheduler.scheduler.slot_allocator.linear_slot_allocator import (
    LinearSlotAllocator,
)
from rl_scheduler.scheduler.operation import OperationInstance, OperationTemplate
from rl_scheduler.scheduler.job import JobInstance, JobTemplate
from rl_scheduler.scheduler.profit import ProfitFunction


@pytest.fixture
def machine_instance():
    machine = MagicMock()
    machine.assigned_operations = []
    machine.last_assigned_end_time = 0
    return machine


@pytest.fixture
def job_instance():
    job_template = MagicMock(spec=JobTemplate)
    profit_function = MagicMock(spec=ProfitFunction)
    job = JobInstance(
        job_instance_id=1,
        job_template=job_template,
        color=(1.0, 0.0, 0.0, 1.0),
        profit_fn=profit_function,
    )
    return job


@pytest.fixture
def operation_instance(job_instance):
    operation_template = OperationTemplate(
        operation_template_id=0, job_template_id=0, type_code="A", duration=5
    )
    operation = OperationInstance(
        operation_template=operation_template,
        predecessor=None,
        successor=None,
        earliest_start_time=10,
    )
    operation.set_job_instance(job_instance)
    job_instance.set_operation_instance_sequence([operation])
    return operation


def test_find_slot_no_operations(machine_instance, operation_instance):
    slot, index = LinearSlotAllocator.find_slot(machine_instance, operation_instance)
    assert slot["start_time"] == 10
    assert slot["end_time"] == 15
    assert index == 0


def test_find_slot_before_first_operation(machine_instance, operation_instance):
    first_op = MagicMock()
    first_op.start_time = 20
    machine_instance.assigned_operations = [first_op]

    slot, index = LinearSlotAllocator.find_slot(machine_instance, operation_instance)
    assert slot["start_time"] == 10
    assert slot["end_time"] == 15
    assert index == 0


def test_find_slot_between_operations(machine_instance, operation_instance):
    first_op = MagicMock()
    first_op.start_time = 10
    first_op.end_time = 15

    second_op = MagicMock()
    second_op.start_time = 25

    machine_instance.assigned_operations = [first_op, second_op]
    operation_instance.earliest_start_time = 15
    operation_instance.duration = 5

    slot, index = LinearSlotAllocator.find_slot(machine_instance, operation_instance)
    assert slot["start_time"] == 15
    assert slot["end_time"] == 20
    assert index == 1


def test_find_slot_after_last_operation(machine_instance, operation_instance):
    last_op = MagicMock()
    last_op.start_time = 0
    last_op.end_time = 30
    machine_instance.assigned_operations = [last_op]

    slot, index = LinearSlotAllocator.find_slot(machine_instance, operation_instance)
    assert slot["start_time"] == 30
    assert slot["end_time"] == 35
    assert index == 1


def test_allocate_operation(machine_instance, operation_instance):
    slot = {"start_time": 10, "end_time": 15}
    insert_index = 0

    LinearSlotAllocator.allocate_operation(
        machine_instance, operation_instance, slot, insert_index
    )

    # Assert that the attributes were set correctly
    assert operation_instance.start_time == 10
    assert operation_instance.end_time == 15
    assert operation_instance.processing_machine == machine_instance
    assert machine_instance.assigned_operations[0] == operation_instance
    assert operation_instance.job_instance.next_op_idx == 1
