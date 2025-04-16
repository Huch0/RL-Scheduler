from abc import ABC, abstractmethod
from scheduler.Scheduler import Scheduler


class RewardHandler(ABC):
    def __init__(self, scheduler: Scheduler, **kwargs):
        """
        Initialize the RewardHandler with a Scheduler instance.

        Args:
            scheduler (Scheduler): The Scheduler object to base the reward computation on.
            **kwargs: Additional parameters for subclasses.
        """
        self.scheduler = scheduler

    @abstractmethod
    def get_intermediate_reward(self) -> float:
        """
        Compute the reward signal based on the current state of the Scheduler.

        Returns:
            float: The computed reward value.
        """
        pass

    @abstractmethod
    def get_terminal_reward(self) -> float:
        """
        Compute the terminal reward signal when all jobs are completed.

        Returns:
            float: The computed terminal reward value.
        """
        pass
