from abc import ABC, abstractmethod
from rl_scheduler.scheduler.scheduler import Scheduler


class PriorityRule(ABC):
    """
    Base class for all priority rules.
    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    @abstractmethod
    def compute_metrics(self) -> dict[tuple[int, int], float]:
        """
        Return a dict {job_instance_id: metric_value}

        ‑ metric_value는 ‘현재 시점에서 이 인스턴스가 갖는
          우선순위 기준의 실제 값’(ex. 남은 여유시간, 추정 지연,
          예상 이윤 등)을 의미.
        """
        raise NotImplementedError()