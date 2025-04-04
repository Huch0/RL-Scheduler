import json
from pathlib import Path
from typing import List
from scheduler.Machine.MachineTemplate import MachineTemplate
from scheduler.Job.JobTemplate import JobTemplate
from scheduler.Operation.OperationTemplate import OperationTemplate


class TemplateLoader:
    @staticmethod
    def load_machine_templates(file_path: Path) -> List[MachineTemplate]:
        with file_path.open("r") as file:
            data = json.load(file)
        return [MachineTemplate(**machine) for machine in data["machines"]]

    @staticmethod
    def load_job_templates(file_path: Path) -> List[JobTemplate]:
        with file_path.open("r") as file:
            data = json.load(file)
        return [JobTemplate(**job) for job in data["jobs"]]

    @staticmethod
    def load_operation_templates(file_path: Path) -> List[OperationTemplate]:
        with file_path.open("r") as file:
            data = json.load(file)
        return [OperationTemplate(**operation) for operation in data["operations"]]
