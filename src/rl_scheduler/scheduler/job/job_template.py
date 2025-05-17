from typing import List, Dict, Any


class JobTemplate:
    def __init__(
        self,
        job_template_id: int,
        operation_template_sequence: List[int],
        color: str = "#FF0000",
    ):
        self.job_template_id = job_template_id
        self.color = color
        self.operation_template_sequence = operation_template_sequence

    def __str__(self):
        return (
            f"JobTemplate(\n"
            f"\tjob_template_id: {self.job_template_id}\n"
            f"\toperation_template_sequence: {self.operation_template_sequence}\n)"
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """템플릿을 JSON 직렬화 가능한 사전으로 변환합니다."""
        return {
            "job_template_id": self.job_template_id,
            "color": self.color,
            "operation_template_sequence": self.operation_template_sequence
        }
