from .job_type_scope_priority_rule import JobTypeScopedPriorityRule
from typing import List

class ETDPriorityRule(JobTypeScopedPriorityRule):
    def assign_priority(self) -> List[int]:
        """
        각 JobTemplate 그룹 내에서
        1) completed=False 인 JobInstance 중
        2) 첫 미완료 OperationInstance의 '최적 종료 시간'을 추정하여
        3) (estimated_finish_time - deadline) 값이 가장 작은 인스턴스의 ID 반환
        모든 인스턴스가 완료되었다면 -1 반환.
        """
        def estimate_finish(machine, op) -> int:
            slots = sorted(
                [(o.start_time, o.end_time) for o in machine.assigned_operations
                 if o.start_time is not None and o.end_time is not None],
                key=lambda x: x[0]
            )
            earliest = op.earliest_start_time
            dur = op.duration
            prev_end = 0
            for (s, e) in slots:
                start_time = max(prev_end, earliest)
                if start_time + dur <= s:
                    return start_time + dur
                prev_end = max(prev_end, e)
            start_time = max(prev_end, earliest)
            return start_time + dur

        result: List[int] = []
        for job_group in self.scheduler.job_instances:
            unfinished = [job for job in job_group if not job.completed]
            if not unfinished:
                result.append(-1)
                continue
            etd_list = []
            for job in unfinished:
                remaining = [op for op in job.operation_instance_sequence if op.end_time is None]
                if not remaining:
                    priority_score = float('inf')
                else:
                    op0 = remaining[0]
                    finish_times = [
                        estimate_finish(machine, op0)
                        for machine in self.scheduler.machine_instances
                        if op0.type_code in machine.machine_template.supported_operation_type_codes
                    ]
                    if finish_times:
                        approx_finish = int(sum(finish_times) / len(finish_times))
                    else:
                        approx_finish = float('inf')
                    total_dur = sum(op.duration for op in job.operation_instance_sequence)
                    rem_durs = [op.duration for op in remaining[1:]]
                    progress_scale = (total_dur - sum(rem_durs)) / total_dur if total_dur > 0 else 0
                    raw_tard = approx_finish - job.profit_fn.deadline
                    priority_score = raw_tard * progress_scale
                etd_list.append((job.job_instance_id, priority_score))
            etd_list.sort(key=lambda x: -x[1])
            result.append(etd_list[0][0])
        return result
