from ..Job.JobInstance import JobTemplate


class OperationTemplate:
    def __init__(
        self,
        operation_template_id: int,
        type_code: str,
        duration: int,
        job_template: JobTemplate,
        predecessor: int, #이렇게 해도 None이 들어갈 수 있나?  검증 필요
    ):
        self.operation_template_id = operation_template_id
        self.type_code = type_code
        self.duration = duration
        self.job_template = job_template
        self.predecessor = predecessor