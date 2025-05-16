"""ETD Priority Rule – paper‑exact formula

T̂_ij = ( Σ_{k=1..k'} p_ijk  / Σ_{k=1..Ki} p_ijk ) * ( Ĉ_ijk' − D_ij )

where
    k'      : index of the *earliest waiting* operation (first unfinished)
    p_ijk   : processing time of O_ijk
    D_ij    : deadline of JobInstance J_ij
    Ĉ_ijk'  : mean finish time by virtually assigning the operation to all
              feasible machines.

Returned metrics: {(template_id, instance_id): etd_value}
Completed jobs → NaN  (for easy filtering in downstream)
"""
from __future__ import annotations

import math
from statistics import mean
from typing import Dict, List, Tuple

from .job_type_scope_priority_rule import JobTypeScopedPriorityRule
from .utils import mean_first_op_finish  # helper already shared

__all__ = ["ETDPriorityRule"]


class ETDPriorityRule(JobTypeScopedPriorityRule):
    """Estimated‑Tardiness‑Driven priority rule (paper‑exact)."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _etd_score(self, job) -> float:
        """Compute ETD value for a single job instance."""
        # 완료된 작업은 metric 계산 불필요 → NaN
        if job.completed:
            return math.nan

        # 남은 operation 인덱스 리스트
        ops = job.operation_instance_sequence
        remaining = [op for op in ops if op.end_time is None]

        # 아래는 불필요하지만 코드 구현상 completed 처리 실수를 방지하기 위해 남겨둠
        if not remaining:
            return math.nan  

        op0 = remaining[0]  # O_ijk' (earliest waiting op)

        c_hat = mean_first_op_finish(op0, self.scheduler.machine_instances)
        if not math.isfinite(c_hat):
            return math.nan  # 어떤 머신에서도 처리 불가

        # duration 분모 (Σ p_ijk)
        total_dur = sum(op.duration for op in ops)
        # duration 분자 (Σ p_ijk, k=1..k') → 이미 끝난 + op0.duration
        done_dur = sum(op.duration for op in ops if op.end_time is not None)
        numer_dur = done_dur + op0.duration
        coeff = numer_dur / total_dur if total_dur else 0.0

        lateness = c_hat - job.profit_fn.deadline
        return coeff * lateness

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def compute_metrics(self) -> Dict[Tuple[int, int], float]:
        """{(template_id, instance_id): ETD score}"""
        return {
            (tid, j.job_instance_id): self._etd_score(j)
            for tid, group in enumerate(self.scheduler.job_instances)
            for j in group
        }

    def assign_priority(self) -> List[int]:
        """Select instance with *largest ETD* (most critical) per template, nan은 제외, 모두 nan 시 -1 반환."""
        metrics = self.compute_metrics()
        result: List[int] = []
        for tid, group in enumerate(self.scheduler.job_instances):
            unfinished = [
                (j.job_instance_id, metrics[(tid, j.job_instance_id)])
                for j in group if not j.completed
            ]
            valid = [(jid, m) for jid, m in unfinished if not math.isnan(m)]
            if not valid:
                result.append(-1)
            else:
                chosen = max(valid, key=lambda x: x[1])[0]
                result.append(chosen)
        return result
