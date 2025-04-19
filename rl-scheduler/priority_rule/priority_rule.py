from abc import ABC
from scheduler.scheduler import Scheduler


class PriorityRule(ABC):
    """
    Base class for all priority rules.
    """
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
