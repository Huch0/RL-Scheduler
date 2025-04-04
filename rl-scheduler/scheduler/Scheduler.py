from typing import List
from Factory import InstanceFactory, TemplateLoader
from Job import JobTemplate, JobInstance
from Machine import MachineTemplate, MachineInstance
from Operation import OperationTemplate, OperationInstance

class Scheduler:
    def __init__(self, machine_config, job_config, operation_config):
        
        # TemplateLoader를 통해서 템플릿을 로드
        self.template_loader = TemplateLoader()
        self.machine_templates = self.template_loader.load_machine_templates(machine_config)
        self.operation_templates = self.template_loader.load_operation_templates(operation_config)
        self.job_templates = self.template_loader.load_job_templates(job_config)
        
        # InstanceFactory 초기화
        self.instance_factory = InstanceFactory(
            machine_templates=self.machine_templates,
            operation_templates=self.operation_templates,
            job_templates=self.job_templates,
        )
        # 인스턴스는 reset()에서 생성
        self.machine_instances : List[MachineInstance] = None
        self.job_instances : List[JobInstance] = None

    def reset(self, repetitions, profit_functions):
        # contract 모듈

        # 반복 수, 가격, 마감일, 지각 패널티를 설정
        # repetitions = [1, 2, 3]
        
        # self.prices = prices
        # self.deadlines = deadlines
        # self.late_penalty = late_penalty

        # 만약 repetitions이 분포 형태로 들어오면 샘플링할것
        # prices ... 는 profit_function 형태로 들어오면 샘플링할것


        self.machine_instances = self.instance_factory.get_new_machine_instances()
        self.job_instances = self.instance_factory.get_new_job_instances(repetitions=repetitions, profit_functions=profit_functions)

    def step(self, chosen_machine, chosen_job, chosen_repetition):
        # M X J X R 형태의 action을 받는다.

        # Step 2: 해당 job의 operation instance 검색
        chosen_op = self.find_op_instance_by_action(chosen_job, chosen_repetition)
        if chosen_op is None:
            raise ValueError("Operation instance not found for job {} repetition {}.".format(chosen_job, chosen_repetition))
        
        # Step 3: 제약 조건 체크 (action mask 모듈을 활용할 수도 있음)
        if not self.check_constraint(chosen_machine, chosen_op):
            raise ValueError("Constraint check failed for machine {} and operation.".format(chosen_machine))
        
        # Step 4: 슬롯 조회 및 할당
        slot = self.find_slot_in_machine(chosen_machine, chosen_op)
        self.allocate_op_to_machine(chosen_machine, chosen_op, slot)
        
        # Step 5: 후처리 (예: 다음 작업의 시작 시간을 업데이트)
        # ...post processing code...

       

    def find_op_instance_by_action(self, chosen_job, chosen_repetition):
        # job instances가 몇 번째 반복인지를 기준으로 오름차순 정렬되어있다 가정
        job_instance = self.job_instances[chosen_job][chosen_repetition]
        
        for operation_instance in job_instance.operation_instance_sequence:
            # 아직 할당되지 않은 operation_instance 중 첫 번째로 찾은 것을 반환한다.
            # 이 부분 job_instance의 멤버 변수에 operation_pointer 변수 추가해서 개선하고 싶다.
            if operation_instance.end_time is None:
                return operation_instance
        # 만약 해당 작업이 다 끝난 상태라면 None을 반환
        # (혹은 예외를 발생시킬 수도 있음)
        return None

    def check_constraint(self, machine_instance, operation_instance):
        # 기계 처리 능력과 operation의 유형이 일치하는지 확인
        return operation_instance.operation_template.operation_type_code in machine_instance.machine_template.supported_operation_type_codes
    
    # TODO
    def find_slot_in_machine(self, machine_instance, operation_instance):
        pass

    # TODO
    def allocate_op_to_machine(self, machine_instance, operation_instance):
        pass