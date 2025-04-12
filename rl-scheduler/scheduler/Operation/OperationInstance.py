from __future__ import annotations
from .OperationTemplate import OperationTemplate
from typing import Optional


class OperationInstance:
    def __init__(
        self,
        # operation_instance_id: int,
        operation_template: OperationTemplate,
        predecessor: Optional[OperationInstance],
        successor: Optional[OperationInstance],
        earliest_start_time: int = 0,
        # job_instance: JobInstance,
    ):
        # self.operation_instance_id = operation_instance_id
        self.operation_template = operation_template

        # Cache frequently accessed attributes from the template
        self.type_code = operation_template.type_code
        self.duration = operation_template.duration

        self.predecessor = predecessor
        self.successor = successor
        self.earliest_start_time = earliest_start_time

        self.job_instance = None  # job_instance와 상호 참조라서 나중에 설정해줘야함

        # Runtime attributes
        self.start_time = None
        self.end_time = None
        self.processing_machine = None

    def set_job_instance(self, job_instance: "JobInstance"):  # noqa: F821
        self.job_instance = job_instance

    def __str__(self):
        pred = "None" if self.predecessor is None else f"{id(self.predecessor)}"
        return f"OperationInstance(id={self.operation_template.operation_template_id}, predecessor={pred}, start={self.start_time}, end={self.end_time})"
