from .OperationTemplate import OperationTemplate
from typing import Optional
from ..Job.JobInstance import JobInstance


class OperationInstance:
    def __init__(
        self,
        operation_instance_id: int,
        operation_template: OperationTemplate,
        predecessor: Optional["OperationInstance"],
        job_instance: JobInstance,
    ):
        self.operation_instance_id = operation_instance_id
        self.operation_template = operation_template
        self.job_instance = job_instance
        self.predecessor = predecessor

        # Runtime attributes
        self.start_time = None
        self.end_time = None
        self.processing_machine = None
