from typing import List


class MachineTemplate:
    def __init__(
        self, machine_template_id: int, supported_operation_type_codes: List[str]
    ):
        self.machine_template_id = machine_template_id
        self.supported_operation_type_codes = supported_operation_type_codes

    def __str__(self):
        return (
            f"MachineTemplate(\n"
            f"\tmachine_template_id: {self.machine_template_id}\n"
            f"\tsupported_operation_type_codes: {self.supported_operation_type_codes}\n)"
        )
