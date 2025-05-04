from typing import List
from ..priority_rule import PriorityRule
from abc import abstractmethod


class JobTypeScopedPriorityRule(PriorityRule):
    """
    각 JobTemplate 내에서 우선순위 높은 JobInstance 선택
    """

    @abstractmethod
    def assign_priority(self) -> List[int]:
        """
        Determine the highest-priority JobInstance ID for each job template group.
        Returns:
            List of selected job_instance_id per job template.
        """
        raise NotImplementedError()
