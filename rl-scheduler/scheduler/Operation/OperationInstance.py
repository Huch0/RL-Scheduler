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

        return f"""OperationInstance(id=
        {self.operation_template.operation_template_id}, predecessor={pred},
        start={self.start_time}, end={self.end_time})"""

    def schedule(self, machine_instance, start_time, end_time):
        """
        Schedules the operation instance on a specific machine and time slot.

        Args:
            machine_instance (MachineInstance): The machine to schedule the operation on.
            start_time (int): The start time of the operation.
            end_time (int): The end time of the operation.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.processing_machine = machine_instance

        # Update the machine's last assigned end time
        machine_instance.last_assigned_end_time = max(
            machine_instance.last_assigned_end_time, end_time
        )

        # If the operation is the last in the sequence, mark the job as completed
        if self.successor is None:
            self.job_instance.completed = True
