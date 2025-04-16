from typing import List
from scheduler.job import JobInstance, JobTemplate
from scheduler.machine import MachineInstance, MachineTemplate
from scheduler.operation import OperationInstance, OperationTemplate
from scheduler.profit import ProfitFunction


# instance를 생성하는 factory
class InstanceFactory:
    def __init__(
        self,
        machine_templates: List[MachineTemplate],
        operation_templates: List[OperationTemplate],
        job_templates: List[JobTemplate],
    ) -> None:
        self.machine_templates = machine_templates
        self.operation_templates = operation_templates
        self.job_templates = job_templates

    # machine instance 생성 메서드
    def get_new_machine_instances(self) -> List[MachineInstance]:
        machine_instances = []
        for machine_template in self.machine_templates:
            machine_instance = MachineInstance(
                machine_template=machine_template,
                assigned_operations=[],  # 나중에 Heap으로 변경해서 사용할 것
            )
            machine_instances.append(machine_instance)
        return machine_instances

    # job instance 생성 메서드
    def get_new_job_instances(
        self,
        repetitions: List[int],
        profit_fn: List[List[ProfitFunction]],
    ) -> List[List[JobInstance]]:

        # operation instance job_template으로 만드는 메서드 (predecessor 연결)
        def create_operation_instances_by_job_template(
            job_template: JobTemplate,
        ) -> List[OperationInstance]:
            operations = []
            predecessor = None
            for operation_template_id in job_template.operation_template_sequence:
                operation_template = self.operation_templates[operation_template_id]
                op_instance = OperationInstance(operation_template, predecessor, None)
                if predecessor:
                    predecessor.successor = op_instance
                operations.append(op_instance)
                predecessor = op_instance
            return operations

        job_instances = []
        for job_template in self.job_templates:
            job_template_id = job_template.job_template_id
            r = repetitions[job_template_id]
            job_type = []
            for i in range(r):
                job_instance = JobInstance(
                    job_instance_id=i,
                    job_template=job_template,
                    profit_fn=profit_fn[job_template_id][
                        i
                    ],  # profit_fn을 job_template_id와 i로 인덱싱하여 가져옴
                )
                operation_sequence = create_operation_instances_by_job_template(
                    job_template
                )
                job_instance.set_operation_instance_sequence(operation_sequence)
                job_type.append(job_instance)
            job_instances.append(job_type)
        return job_instances
