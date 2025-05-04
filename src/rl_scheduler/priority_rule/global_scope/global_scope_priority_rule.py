from ..priority_rule import PriorityRule
from abc import abstractmethod


class GlobalScopedPriorityRule(PriorityRule):
    """
    전체 스케줄링 대상 중 가장 우선순위 높은 작업 선택
    """

    @abstractmethod
    def select_priority(self) -> tuple[int, int]:
        """
        Determine the highest-priority job (template ID) and repetition
        index across all job instances.
        Returns:
            Tuple of (job_id, repetition_index), or (-1, -1) if none available.
        """
        raise NotImplementedError()
