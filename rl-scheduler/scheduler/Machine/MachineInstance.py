from .MachineTemplate import MachineTemplate
from typing import List
from ..Operation.OperationInstance import OperationInstance


class MachineInstance:
    def __init__(
        self,
        machine_template: MachineTemplate,
        assigned_operations: List[OperationInstance],
    ):
        self.machine_template = machine_template
        self.assigned_operations = assigned_operations
