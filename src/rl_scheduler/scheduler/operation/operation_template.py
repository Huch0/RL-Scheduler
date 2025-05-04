class OperationTemplate:
    def __init__(
        self,
        operation_template_id: int,
        type_code: str,
        duration: int,
        job_template_id: int,
    ):
        self.operation_template_id = operation_template_id
        self.type_code = type_code
        self.duration = duration
        self.job_template_id = job_template_id

    def __str__(self):
        return (
            f"OperationTemplate(\n"
            f"\toperation_template_id: {self.operation_template_id}\n"
            f"\ttype_code: {self.type_code}\n"
            f"\tduration: {self.duration}\n"
            f"\tjob_template_id: {self.job_template_id}\n)"
        )
