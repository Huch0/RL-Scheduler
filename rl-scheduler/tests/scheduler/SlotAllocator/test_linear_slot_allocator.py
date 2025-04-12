import pytest
from unittest.mock import MagicMock
from scheduler.SlotAllocator.linear_slot_allocator import LinearSlotAllocator


@pytest.fixture
def machine_instance():
    machine = MagicMock()
    machine.assigned_operations = []
    return machine


@pytest.fixture
def operation_instance():
    operation = MagicMock()
    operation.earliest_start_time = 10
    operation.duration = 5
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

    assert operation_instance.start_time == 10
    assert operation_instance.end_time == 15
    assert operation_instance.processing_machine == machine_instance
    assert machine_instance.assigned_operations[0] == operation_instance
