from __future__ import annotations

from .job_template import JobTemplate
from ..operation.operation_instance import OperationInstance
from ..profit import ProfitFunction
from typing import List


class JobInstance:
    def __init__(
        self,
        job_instance_id: int,
        job_template: JobTemplate,
        profit_fn: ProfitFunction,
    ):
        self.job_instance_id = job_instance_id
        self.job_template = job_template
        self.profit_fn = profit_fn
        self.operation_instance_sequence = None
        self.completed = False  # Tracks whether the job is completed

    def set_operation_instance_sequence(
        self, operation_instance_sequence: List[OperationInstance]
    ):
        self.operation_instance_sequence = operation_instance_sequence
        for operation_instance in self.operation_instance_sequence:
            operation_instance.set_job_instance(self)

    def __str__(self):
        template_id = getattr(self.job_template, "job_template_id", "N/A")
        profit = f"price={self.profit_fn.price}" if self.profit_fn else "NoProfit"
        ops = (
            len(self.operation_instance_sequence)
            if self.operation_instance_sequence
            else 0
        )
        return f"""JobInstance(id={self.job_instance_id}, template_id=
        {template_id}, {profit}, ops_count={ops})"""
