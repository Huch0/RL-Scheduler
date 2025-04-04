from typing import List
from scheduler.Job import JobInstance, JobTemplate
from scheduler.Machine import MachineInstance, MachineTemplate
from scheduler.Operation import OperationInstance, OperationTemplate


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
                assigned_operations=[]  # 나중에 Heap으로 변경해서 사용할 것
            )
            machine_instances.append(machine_instance)
        return machine_instances

   

    # job instance 생성 메서드
    def get_new_job_instances(
        self,
        repetitions: List[int],
        prices: List[int],
        deadlines: List[int],
        late_penalty: dict[int],
    ) -> List[List[JobInstance]]:
        
         # operation instance job_template으로 만드는 메서드 (predecessor 연결)
        def create_operation_instances_by_job_template(job_template: JobTemplate) -> List[OperationInstance]:
            operations = []
            predecessor = None
            # 현재 아래 부분이 최근 추가 사항과 맞지 않음.
            for operation_template in job_template.operation_template_sequence:
                op_instance = OperationInstance(operation_template, predecessor)
                operations.append(op_instance)
                predecessor = op_instance
            return operations
        
        job_instances = []
        for job_template in self.job_templates:
            for r in repetitions:
                job_type = []
                for i in range(r):
                    job_instance = JobInstance(
                        # job instance id 형식은 1이상으로 설정 (몇번째 반복인지 의미)
                        job_instance_id=i + 1, 
                        job_template=job_template,
                        earnings=prices[i],
                        deadline=deadlines[i],
                        late_penalty=late_penalty[i],
                    )
                    operation_seqeunce = create_operation_instances_by_job_template(job_template)
                    job_instance.set_operation_instance_sequence(operation_seqeunce)
                    job_type.append(job_instance)
                job_instances.append(job_type)
        return job_instances