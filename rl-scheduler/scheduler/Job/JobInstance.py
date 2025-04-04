from __future__ import annotations

from .JobTemplate import JobTemplate
from ..Operation.OperationInstance import OperationInstance
from typing import List


class JobInstance:
    def __init__(
        self,
        job_instance_id: int,
        job_template: JobTemplate,
        # operation_instance_sequence: List[OperationInstance],
        deadline: int,
        earnings: int,
        late_penalty: int,
    ):
        self.job_instance_id = job_instance_id
        self.job_template = job_template

        self.deadline = deadline
        self.earnings = earnings
        self.late_penalty = late_penalty

        # 상호참조라 우선 None
        self.operation_instance_sequence = None

    def set_operation_instance_sequence(
        self, operation_instance_sequence: List[OperationInstance]
    ):
        self.operation_instance_sequence = operation_instance_sequence
        for operation_instance in self.operation_instance_sequence:
            operation_instance.set_job_instance(self)
