from .job_type_scope_priority_rule import JobTypeScopedPriorityRule
from typing import List
import math

class EDDPriorityRule(JobTypeScopedPriorityRule):
    def assign_priority(self) -> List[int]:
        """
        각 JobTemplate 그룹 내에서 완료되지 않은 인스턴스 중 가장 작은 metric을 갖는 인스턴스를 선택.
        모든 metric이 nan이거나 완료된 경우 -1 반환.
        """
        metrics = self.compute_metrics()
        result: List[int] = []
        for jt_id, group in enumerate(self.scheduler.job_instances):
            unfinished = [
                (job.job_instance_id, metrics[(jt_id, job.job_instance_id)])
                for job in group if not getattr(job, 'completed', False)
            ]
            # nan인 metric 제외
            valid = [(jid, m) for jid, m in unfinished if not math.isnan(m)]
            if not valid:
                result.append(-1)
            else:
                chosen_id = min(valid, key=lambda x: x[1])[0]
                result.append(chosen_id)
        return result

    def compute_metrics(self) -> dict[tuple[int, int], float]:
        """
            Return (job_template_id, instance_id) → deadline slack
        """
        metrics = {}
        for jt_id, group in enumerate(self.scheduler.job_instances):
            for job in group:
                if job.completed:
                    metrics[(jt_id, job.job_instance_id)] = math.nan
                    continue
                dline = job.profit_fn.deadline if job.profit_fn else float('inf')
                metrics[(jt_id, job.job_instance_id)] = dline if math.isfinite(dline) else math.nan
        return metrics