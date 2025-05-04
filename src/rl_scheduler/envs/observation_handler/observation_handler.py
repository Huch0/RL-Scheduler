from abc import ABC, abstractmethod
from gymnasium import spaces
from rl_scheduler.scheduler.scheduler import Scheduler
from gymnasium.core import ObsType


class ObservationHandler(ABC):
    def __init__(self, scheduler: Scheduler, **kwargs):
        """
        Initialize the ObservationHandler with a Scheduler instance.

        Args:
            scheduler (Scheduler): The Scheduler object to base the observation space on.
            **kwargs: Additional parameters for subclasses.
        """
        self.scheduler = scheduler
        self.observation_space = self.create_observation_space()

    @abstractmethod
    def create_observation_space(self) -> spaces.Space:
        """
        Create a gymnasium observation space based on the Scheduler's configuration.

        Returns:
            spaces.Space: A gymnasium space representing the observation space.
        """
        pass

    @abstractmethod
    def get_observation(self) -> ObsType:
        """
        Generate the current observation of the environment using the Scheduler.

        Returns:
            ObsType: The current state observation.
        """
        pass
