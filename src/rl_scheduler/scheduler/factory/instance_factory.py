from typing import List
from rl_scheduler.scheduler.job import JobInstance, JobTemplate
from rl_scheduler.scheduler.machine import MachineInstance, MachineTemplate
from rl_scheduler.scheduler.operation import OperationInstance, OperationTemplate
from rl_scheduler.scheduler.profit import ProfitFunction
import matplotlib.colors as mcolors


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
            repeat = repetitions[job_template_id]

            # Compute an alpha (opacity) value for each repetition such that
            # the first instance has the lowest opacity (0.4) and the last
            # reaches 1.0, distributed uniformly across `r` repetitions.
            base_alpha = 0.4
            max_alpha = 1.0
            if repeat > 1:
                alpha_step = (max_alpha - base_alpha) / (repeat - 1)
            else:
                alpha_step = 0.0  # single instance ⇒ full opacity

            job_type = []

            # ------------------------------------------------------------------ #
            # Color handling: inherit RGB from template, but increase opacity
            # (alpha channel) as job_instance_id grows so later instances
            # appear more vivid in the plot.
            # ------------------------------------------------------------------ #
            base_color = job_template.color  # '#RRGGBB'

            # Convert to RGB tuple in 0‑1 range
            r, g, b = mcolors.to_rgb(base_color)

            for i in range(repeat):
                # Alpha increases uniformly with repetition index
                alpha = min(max_alpha, base_alpha + i * alpha_step)
                job_instance = JobInstance(
                    job_instance_id=i,
                    job_template=job_template,
                    color=(r, g, b, alpha),  # RGBA tuple
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
