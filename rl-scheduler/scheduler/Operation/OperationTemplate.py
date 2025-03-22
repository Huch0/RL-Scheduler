from ..Job.JobInstance import JobTemplate


class OperationTemplate:
    def __init__(
        self,
        operation_template_id: int,
        type_code: str,
        duration: int,
        job_template: JobTemplate,
    ):
        self.operation_template_id = operation_template_id
        self.type_code = type_code
        self.duration = duration
        self.job_template = job_template
