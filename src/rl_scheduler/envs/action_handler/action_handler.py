from abc import ABC, abstractmethod
from gymnasium import spaces
from rl_scheduler.scheduler.scheduler import Scheduler
import numpy as np


class ActionHandler(ABC):
    def __init__(self, scheduler: Scheduler, **kwargs):
        """
        Initialize the ActionSpaceHandler with a Scheduler instance.

        Args:
            scheduler (Scheduler): The Scheduler object to base the action space on.
            **kwargs: Additional parameters for subclasses.
        """
        self.scheduler = scheduler
        self.action_space = self.create_action_space()

    @abstractmethod
    def create_action_space(self) -> spaces.Space:
        """
        Create a gymnasium action space based on the Scheduler's configuration.

        Returns:
            spaces.Space: A gymnasium space representing the action space.
        """
        pass

    @abstractmethod
    def convert_action(self, action) -> tuple:
        """
        Convert an action from the action space into a compatible form with Scheduler

        Args:
            action: The action to convert.

        Returns:
            tuple: A converted action.
        """
        pass

    @abstractmethod
    def compute_action_mask(self) -> "np.ndarray":
        """
        Compute the action mask for the current state.

        Returns:
            np.ndarray: A boolean array indicating valid actions.
        """
        pass
