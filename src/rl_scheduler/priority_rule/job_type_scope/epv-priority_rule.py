# epv_priority_rule.py
import math
from typing import Dict, List, Tuple

from .job_type_scope_priority_rule import JobTypeScopedPriorityRule
from .utils import estimated_job_finish 

class EPVPriorityRule(JobTypeScopedPriorityRule):
    """
    EPV = Estimated Profit Value
    예상 이윤(profit)이 높은 인스턴스를 우선 선택한다.
    profit = price − late_penalty * max(0, estimated_finish − deadline)
    """

    # ------------------------- internal helpers -------------------------
    def _profit_score(self, job) -> float:
        if job.completed:
            return math.nan
        pf = job.profit_fn
        if pf is None:
            return -float("inf")

        est_finish = estimated_job_finish(job, self.scheduler.machine_instances)
        if not math.isfinite(est_finish):
            return math.nan

        lateness = max(0, est_finish - pf.deadline)
        return pf.price - pf.late_penalty * lateness

    # ------------------------- public API -------------------------------
    def compute_metrics(self) -> Dict[Tuple[int, int], float]:
        raw = {
            (tid, job.job_instance_id): self._profit_score(job)
            for tid, group in enumerate(self.scheduler.job_instances)
            for job in group
        }
        # inf을 nan으로 통일
        return {k: (v if math.isfinite(v) else math.nan) for k, v in raw.items()}

    def assign_priority(self) -> List[int]:
        """
        metric이 nan인 인스턴스 제외 후 profit이 가장 큰 인스턴스 선택, 모두 nan 시 -1.
        """
        metrics = self.compute_metrics()
        result: List[int] = []
        for tid, group in enumerate(self.scheduler.job_instances):
            unfinished = [
                (job.job_instance_id, metrics[(tid, job.job_instance_id)])
                for job in group if not job.completed
            ]
            valid = [(jid, m) for jid, m in unfinished if not math.isnan(m)]
            if not valid:
                result.append(-1)
            else:
                result.append(max(valid, key=lambda x: x[1])[0])
        return result
