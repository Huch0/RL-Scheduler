from abc import ABC, abstractmethod
from typing import List
from scheduler.scheduler import Scheduler


class PriorityRule(ABC):
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    @abstractmethod
    def assign_priority(self) -> List[int]:
        """
        각 JobTemplate 내에서 우선순위가 높은 JobInstance의 id를 결정함.
        Returns:
            job_instance_id의 리스트 (예: [0, 1, 0, 2, 1])
        """
        raise NotImplementedError("You should implement this method.")