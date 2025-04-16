from .action_handler import ActionHandler
from gymnasium import spaces
from scheduler.Scheduler import Scheduler


class MJRHandler(ActionHandler):
    def __init__(self, scheduler: Scheduler, max_repetition: int):
        """
        Initialize the MJRHandler with a Scheduler instance and max_repetition.

        Args:
            scheduler (Scheduler): The Scheduler object to base the action space on.
            max_repetition (int): The maximum number of repetitions.
        """
        self.max_repetition = max_repetition
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
                self.max_repetition,  # Max repetitions (R)
            ]
        )

    def convert_action(self, action) -> tuple:
        """
        Convert an action from the action space into M x J x R form.

        Args:
            action: The action to convert.

        Returns:
            tuple: A tuple (M, J, R) representing the machine, job, and repetition.
        """
        return tuple(action)
