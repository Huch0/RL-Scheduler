from ..Operation.OperationTemplate import OperationTemplate
from typing import List


class JobTemplate:
    def __init__(
        self, job_template_id: int, operation_template_sequence: List[OperationTemplate]
    ):
        self.job_template_id = job_template_id
        self.operation_template_sequence = operation_template_sequence
