from abc import ABC, abstractmethod
from rl_scheduler.scheduler import Scheduler


class Renderer(ABC):
    @staticmethod
    @abstractmethod
    def render(scheduler: Scheduler, title: str = "Renderer"):
        raise NotImplementedError("You should implement this method.")
