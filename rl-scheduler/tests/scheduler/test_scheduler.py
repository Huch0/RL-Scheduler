import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from scheduler.Scheduler import Scheduler
from scheduler.SlotAllocator import LinearSlotAllocator


@pytest.fixture
def mock_instance_factory():
    factory = MagicMock()
    factory.get_new_machine_instances.return_value = [MagicMock()]
    factory.get_new_job_instances.return_value = [[MagicMock()]]
    return factory


@pytest.fixture
def mock_template_loader():
    with patch("scheduler.Scheduler.TemplateLoader") as mock_loader:
        mock_loader.load_machine_templates.return_value = []
        mock_loader.load_operation_templates.return_value = []
        mock_loader.load_job_templates.return_value = []
        yield mock_loader


@pytest.fixture
def scheduler(mock_instance_factory, mock_template_loader):
    with patch(
        "scheduler.Scheduler.InstanceFactory", return_value=mock_instance_factory
    ):
        return Scheduler(
            machine_config_path=Path("machine_config.json"),
            job_config_path=Path("job_config.json"),
            operation_config_path=Path("operation_config.json"),
        )


def test_scheduler_initialization(scheduler):
    assert scheduler.slot_allocator == LinearSlotAllocator
    assert scheduler.machine_templates == []
    assert scheduler.operation_templates == []
    assert scheduler.job_templates == []


def test_scheduler_reset(scheduler, mock_instance_factory):
    repetitions = [1, 2, 3]
    profit_functions = [MagicMock()]
    scheduler.reset(repetitions, profit_functions)

    assert (
        scheduler.machine_instances
        == mock_instance_factory.get_new_machine_instances.return_value
    )
    assert (
        scheduler.job_instances
        == mock_instance_factory.get_new_job_instances.return_value
    )
    mock_instance_factory.get_new_job_instances.assert_called_once_with(
        repetitions=repetitions, profit_functions=profit_functions
    )


def test_scheduler_step_valid_action(scheduler):
    scheduler.machine_instances = [MagicMock()]
    scheduler.job_instances = [[MagicMock()]]
    chosen_machine = scheduler.machine_instances[0]
    chosen_job = scheduler.job_instances[0][0]
    chosen_op = MagicMock()
    chosen_job.operation_instance_sequence = [chosen_op]
    chosen_op.end_time = None

    with (
        patch.object(
            scheduler, "check_constraint", return_value=True
        ) as mock_check_constraint,
        patch.object(
            scheduler.slot_allocator, "find_and_allocate_slot"
        ) as mock_allocate_slot,
    ):
        scheduler.step(0, 0, 0)

        mock_check_constraint.assert_called_once_with(chosen_machine, chosen_op)
        mock_allocate_slot.assert_called_once_with(
            machine_instance=chosen_machine, operation_instance=chosen_op
        )
        assert chosen_op.successor.earliest_start_time == chosen_op.end_time


def test_scheduler_step_constraint_violation(scheduler):
    scheduler.machine_instances = [MagicMock()]
    scheduler.job_instances = [[MagicMock()]]
    chosen_job = scheduler.job_instances[0][0]
    chosen_op = MagicMock()
    chosen_job.operation_instance_sequence = [chosen_op]
    chosen_op.end_time = None

    with patch.object(scheduler, "check_constraint", return_value=False):
        with pytest.raises(ValueError, match="Constraint check failed"):
            scheduler.step(0, 0, 0)


def test_scheduler_step_no_operation_instance(scheduler):
    scheduler.machine_instances = [MagicMock()]
    scheduler.job_instances = [[MagicMock()]]
    chosen_job = scheduler.job_instances[0][0]
    chosen_job.operation_instance_sequence = []

    with pytest.raises(ValueError, match="Operation instance not found"):
        scheduler.step(0, 0, 0)
