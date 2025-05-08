from typing import List


class JobTemplate:
    def __init__(
        self,
        job_template_id: int,
        operation_template_sequence: List[int],
        color: str = "#FF0000",
    ):
        self.job_template_id = job_template_id
        # self.color = color
        # Red for now, to be replaced with a color scheme
        self.color = "#FF0000"
        self.operation_template_sequence = operation_template_sequence

    def __str__(self):
        return (
            f"JobTemplate(\n"
            f"\tjob_template_id: {self.job_template_id}\n"
            f"\toperation_template_sequence: {self.operation_template_sequence}\n)"
        )
