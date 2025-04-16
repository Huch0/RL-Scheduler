from abc import ABC, abstractmethod
from scheduler.Scheduler import Scheduler


class InfoHandler(ABC):
    def __init__(self, scheduler: Scheduler):
        """
        Initialize the InfoHandler with a Scheduler instance.

        Args:
            scheduler (Scheduler): The Scheduler object to base the info computation on.
        """
        self.scheduler = scheduler

    @abstractmethod
    def get_info(self) -> dict:
        """
        Compute and return an info dictionary based on the current state of the Scheduler.

        Returns:
            dict: A dictionary containing information about the current state of the Scheduler.
        """
        pass
