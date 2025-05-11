from rl_scheduler.scheduler import Scheduler
from rl_scheduler.envs.action_handler.mj_handler import MJHandler
from rl_scheduler.priority_rule.job_type_scope.etd_priority_rule import ETDPriorityRule
from gymnasium import spaces
from unittest.mock import MagicMock


def test_mj_handler_with_real_etd_priority_rule():
    # Setup: two machines, two job templates
    n_machines = 2
    n_jobs = 2

    # Mock Scheduler with necessary attributes
    scheduler = MagicMock(spec=Scheduler)
    scheduler.machine_templates = [MagicMock() for _ in range(n_machines)]
    scheduler.job_templates = [MagicMock() for _ in range(n_jobs)]
    scheduler.machine_instances = [MagicMock() for _ in range(n_machines)]

    # Create job_instances: each job template has two repetitions
    job_instances = []
    for j in range(n_jobs):
        reps = []
        for r in range(2):
            job = MagicMock()
            job.completed = False
            # At least one operation to allow indexing
            job.operation_instance_sequence = [MagicMock()]
            # Simulate priority by setting current_operation_index
            job.current_operation_index = r
            reps.append(job)
        job_instances.append(reps)
    scheduler.job_instances = job_instances

    # Initialize MJHandler with real ETDPriorityRule
    handler = MJHandler(scheduler, priority_rule_id="etd")

    # The action space should be Discrete with size machines * jobs
    action_space = handler.create_action_space()
    assert isinstance(action_space, spaces.Discrete)
    assert action_space.n == n_machines * n_jobs

    # The priority_rule should be an ETDPriorityRule instance
    assert isinstance(handler.priority_rule, ETDPriorityRule)

    # Compute expected repetition indices via the rule directly
    expected_priorities = handler.priority_rule.assign_priority()
    assert isinstance(expected_priorities, list)
    assert len(expected_priorities) == n_jobs

    # Test convert_action returns (machine, job, correct repetition)
    for action in range(n_machines * n_jobs):
        converted = handler.convert_action(action)
        machine_idx = action // n_jobs
        job_idx = action % n_jobs
        assert len(converted) == 3
        assert converted[0] == machine_idx
        assert converted[1] == job_idx
        # Converted repetition must match the rule's output
        assert converted[2] == expected_priorities[job_idx]
