from .JobTemplate import JobTemplate
from ..Operation.OperationInstance import OperationInstance
from typing import List


class JobInstance:
    def __init__(
        self,
        job_instance_id: int,
        job_template: JobTemplate,
        operation_instance_sequence: List[OperationInstance],
        deadline: int,
        earnings: int,
        late_penalty: int,
    ):
        self.job_instance_id = job_instance_id
        self.job_template = job_template
        self.operation_instance_sequence = operation_instance_sequence
        self.deadline = deadline
        self.earnings = earnings
        self.late_penalty = late_penalty
