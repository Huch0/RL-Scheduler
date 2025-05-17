from .machine_template import MachineTemplate
from typing import List
from ..operation.operation_instance import OperationInstance


class MachineInstance:
    def __init__(
        self,
        machine_template: MachineTemplate,
        assigned_operations: List[OperationInstance],
    ):
        self.machine_template = machine_template
        self.supported_operation_type_codes = (
            self.machine_template.supported_operation_type_codes
        )
        self.assigned_operations = assigned_operations
        self.last_assigned_end_time = (
            0  # Tracks the end time of the last assigned operation
        )

    def __str__(self):
        return f"""MachineInstance(template={self.machine_template},
        ops_count={len(self.assigned_operations)})"""
