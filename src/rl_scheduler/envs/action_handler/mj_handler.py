from .action_handler import ActionHandler
from gymnasium import spaces
from rl_scheduler.scheduler import Scheduler
from rl_scheduler.priority_rule import get_job_type_rule


class MJHandler(ActionHandler):
    def __init__(self, scheduler: Scheduler, priority_rule_id: str = "etd"):
        """
        Initialize the MJRHandler with a Scheduler instance and max_repetition.

        Args:
            scheduler (Scheduler): The Scheduler object to base the action
            space on.
            max_repetition (int): The maximum number of repetitions.
        """
        self.priority_rule = get_job_type_rule(priority_rule_id, scheduler)
        super().__init__(scheduler)

    def create_action_space(self) -> spaces.Space:
        """
        Create a gymnasium action space based on the Scheduler's configuration.

        Returns:
            spaces.Space: A gymnasium space representing the action space.
        """
        return spaces.MultiDiscrete(
            [
                len(self.scheduler.machine_templates),  # Number of machines (M)
                len(self.scheduler.job_templates),  # Number of jobs (J)
            ]
        )

    def convert_action(self, action) -> tuple:
        """
        Convert an action from the action space into M x J x R form.

        Args:
            action: The action to convert. (tuple of (M, J))

        Returns:
            tuple: A tuple (M, J, R) representing the machine, job,
            and repetition.
        """

        # Expect `action` to be a tuple: (machine_index, job_template_index)
        try:
            machine_idx, job_idx = action
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid action format {action!r}: expected (machine_idx, " f"job_idx)"
            )

        # Determine repetition index via the priority rule
        # `assign_priority()` returns a list of chosen repetition per job
        # template
        priorities = self.priority_rule.assign_priority()
        try:
            repetition_idx = priorities[job_idx]
        except IndexError:
            raise IndexError(f"Priority rule returned no entry for job index {job_idx}")

        return machine_idx, job_idx, repetition_idx
