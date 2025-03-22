from copy import deepcopy
from typing import List, Tuple
from scheduler.Job import JobInstance, JobTemplate
from scheduler.Machine import MachineInstance, MachineTemplate
from scheduler.Operation import OperationInstance, OperationTemplate

# instance를 생성하는 factory
class InstanceFactory:
    def __init__(self, 
            machine_templates: List[MachineTemplate], 
            operation_templates: List[OperationTemplate], 
            job_templates: List[JobTemplate]
        ) -> None:
        self.machine_templates = machine_templates
        self.operation_templates = operation_templates
        self.job_templates = job_templates

    # scheduler 모듈에서 매 reset마다 호출 필요
    def get_new_instance(
            self, 
            repetitions: List[int], 
            prices: List[int],  
            deadlines: List[int],
            late_penalty: dict[int],
        ) -> dict(List[MachineInstance], List[List[JobInstance]]):
        
        # 만약 deadline을 일일이 주지 않는다면 확률 분포로부터 샘플링해야함
        # 25.03.22. 드는 생각 : price, deadline과 late_panelty를 묶어서 가격 함수로 표현이 가능한데...

        # machine instance 생성
        machine_instances = []
        for machine_template in self.machine_templates:
            machine_instance = MachineInstance(
                machine_template = machine_template,
                assigned_operations = [], #나중에 Heap으로 변경해서 사용할 것
            )
            machine_instances.append(machine_instance)

        # job instance, operation instance 생성
        job_instances = []
        for job_template in self.job_templates:
            for r in repetitions:
                job_type = []
                for i in range(r):       
                    job_instance = JobInstance(
                        job_instance_id = i+1,
                        job_template = job_template,
                        earnings = prices[i],
                        deadline = deadlines[i], 
                        late_penalty = late_penalty[i],                 
                    )
                    job_instance.set_operation_instance_sequence([OperationInstance(operation_template) for operation_template in job_template.operation_template_sequence])
                    job_type.append(job_instance)
                job_instances.append(job_type)

        instance = {
            'machine_instances': machine_instances,
            'job_instances': job_instances,
        }

        return instance