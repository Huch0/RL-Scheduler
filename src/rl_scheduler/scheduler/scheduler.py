from typing import List, Dict, Any, Type
from pathlib import Path
from .factory import InstanceFactory, TemplateLoader
from .job import JobInstance
from .machine import MachineInstance
from .slot_allocator import SlotAllocator, LinearSlotAllocator
from rl_scheduler.graph.sync import SchedulerGraphSync


class Scheduler:
    def __init__(
        self,
        machine_config_path: Path,
        job_config_path: Path,
        operation_config_path: Path,
        slot_allocator: Type[SlotAllocator] = LinearSlotAllocator,
    ):
        self.machine_templates = TemplateLoader.load_machine_templates(
            machine_config_path
        )
        self.operation_templates = TemplateLoader.load_operation_templates(
            operation_config_path
        )
        self.job_templates = TemplateLoader.load_job_templates(job_config_path)

        # InstanceFactory 초기화
        self.instance_factory = InstanceFactory(
            machine_templates=self.machine_templates,
            operation_templates=self.operation_templates,
            job_templates=self.job_templates,
        )
        # 인스턴스는 reset()에서 생성
        self.machine_instances: List[MachineInstance] = None
        self.job_instances: List[List[JobInstance]] = None

        # machine slot allocator Strategy
        self.slot_allocator = slot_allocator

        self.timestep = 0
        self.graph_sync = None  # Initialized in reset

    def reset(self, repetitions, profit_functions):
        self.machine_instances = self.instance_factory.get_new_machine_instances()
        self.job_instances = self.instance_factory.get_new_job_instances(
            repetitions=repetitions, profit_fn=profit_functions
        )
        self.timestep = 0
        # Initialize graph synchronization
        self.graph_sync = SchedulerGraphSync(self)

    def get_templates_as_dicts(self) -> Dict[str, Any]:
        """템플릿들을 JSON 직렬화 가능한 사전 형태로 반환합니다."""
        return {
            "machine_config": [template.to_dict() for template in self.machine_templates],
            "job_config": [template.to_dict() for template in self.job_templates],
            "operation_config": [template.to_dict() for template in self.operation_templates]
        }
        
    def step(self, chosen_machine_id: int, chosen_job_id: int, chosen_repetition: int):
        """
        Executes a single scheduling step based on the provided action.

        Args:
            chosen_machine_id (int): The ID of the machine to allocate.
            chosen_job_id (int): The ID of the job to process.
            chosen_repetition (int): The repetition index of the job.

        Steps:
            1. Receives an action in the form of M x J x R (machine x job x
            Repetition).
            2. Finds the operation instance for the given job and repetition.
               - Raises a ValueError if the operation instance is not found.
            3. Checks constraints between the chosen machine and operation.
               - Raises a ValueError if the constraints are not satisfied.
            4. Allocates a slot for the operation on the machine.
            5. Updates the earliest start time of the successor operation.
        """
        # Find the operation instance for the given job and repetition.
        try:
            chosen_op = self.find_op_instance_by_action(
                chosen_job_id, chosen_repetition
            )
        except IndexError as e:  # The selected job instance is already completed.
            raise ValueError(
                f"Operation instance not found for job {chosen_job_id} "
                f"repetition {chosen_repetition}. Probably all operations are already assigned."
            ) from e

        # Check constraints between the chosen machine and operation.
        chosen_machine = self.machine_instances[chosen_machine_id]
        if not self.check_constraint(chosen_machine, chosen_op):
            raise ValueError(
                f"Constraint check failed for machine {chosen_machine_id} "
                f"and operation."
            )

        # Allocate a slot for the operation on the machine.
        self.slot_allocator.find_and_allocate_slot(
            machine_instance=chosen_machine, operation_instance=chosen_op
        )
        # Notify graph sync of assignment event
        if self.graph_sync:
            self.graph_sync.on_assignment(chosen_machine, chosen_op)

        # Update the earliest start time of the successor operation.
        if chosen_op.successor is not None:
            # Notify graph sync of completion event for previous op
            if self.graph_sync:
                self.graph_sync.on_completion(chosen_op)
            chosen_op.successor.earliest_start_time = chosen_op.end_time

        self.timestep += 1

    def find_op_instance_by_action(self, chosen_job_id: int, chosen_repetition: int):
        # job instances가 몇 번째 반복인지를 기준으로 오름차순 정렬되어있다 가정
        job_instance = self.job_instances[chosen_job_id][chosen_repetition]

        return job_instance.operation_instance_sequence[job_instance.next_op_idx]
        #
        # for operation_instance in job_instance.operation_instance_sequence:
        #     # 아직 할당되지 않은 operation_instance 중 첫 번째로 찾은 것을 반환한다.
        #     # 이 부분 job_instance의 멤버 변수에 operation_pointer 변수 추가해서 개선하고 싶다.
        #     if operation_instance.end_time is None:
        #         return operation_instance
        #
        # raise ValueError(
        #     f"Operation instance not found for job {chosen_job_id} "
        #     f"repetition {chosen_repetition}."
        #     f"Probably all operations are already assigned."
        # )

    def check_constraint(self, machine_instance, operation_instance):
        # 기계 처리 능력과 operation의 유형이 일치하는지 확인
        return (
            operation_instance.type_code
            in machine_instance.machine_template.supported_operation_type_codes
        )

    def is_all_job_instances_scheduled(self):
        """
        Check if all job instances are scheduled.

        Returns:
            bool: True if all job instances are scheduled, False otherwise.
        """

        return all(
            job_instance.completed
            for job_type in self.job_instances
            for job_instance in job_type
        )
