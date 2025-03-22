from .OperationTemplate import OperationTemplate
from typing import Optional
from ..Job.JobInstance import JobInstance


class OperationInstance:
    def __init__(
        self,
        # operation_instance_id: int,
        operation_template: OperationTemplate,
        # predecessor: Optional["OperationInstance"],
        # job_instance: JobInstance,
    ):
        # self.operation_instance_id = operation_instance_id
        self.operation_template = operation_template
        
        # self.predecessor = predecessor

        self.job_instance = None # job_instance와 상호 참조라서 나중에 설정해줘야함

        # Runtime attributes
        self.start_time = None
        self.end_time = None
        self.processing_machine = None

    def set_job_instance(self, job_instance: JobInstance):
        self.job_instance = job_instance
