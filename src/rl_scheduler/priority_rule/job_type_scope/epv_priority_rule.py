# epv_priority_rule.py
import math
from typing import Dict, List, Tuple

from .job_type_scope_priority_rule import JobTypeScopedPriorityRule
from .utils import estimated_job_finish
from .edd_priority_rule import EDDPriorityRule  # tie‐break용

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
        각 JobTemplate 그룹 내에서 EPV(metric)이 가장 작은 인스턴스를 선택.
        metric이 nan이거나 완료된 경우 제외. 동점이면 EDD(metric) 기준으로 선택.
        """
        epv_metrics = self.compute_metrics()
        # EDD metrics for tie-breaking
        edd_metrics = EDDPriorityRule(self.scheduler).compute_metrics()

        result: List[int] = []
        for jt_id, group in enumerate(self.scheduler.job_instances):
            # unfinished & valid(epv not nan)
            valid = [
                (job.job_instance_id, epv_metrics[(jt_id, job.job_instance_id)])
                for job in group
                if not getattr(job, "completed", False)
                  and not math.isnan(epv_metrics[(jt_id, job.job_instance_id)])
            ]
            if not valid:
                result.append(-1)
                continue

            # find min EPV value
            min_epv = min(v for _, v in valid)
            # candidates with same EPV
            cands = [jid for jid, v in valid if v == min_epv]
            if len(cands) == 1:
                result.append(cands[0])
            else:
                # tie-break by smallest EDD (deadline slack)
                chosen = min(cands, key=lambda jid: edd_metrics[(jt_id, jid)])
                result.append(chosen)

        return result
