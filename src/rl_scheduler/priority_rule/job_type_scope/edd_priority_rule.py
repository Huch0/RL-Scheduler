from .job_type_scope_priority_rule import JobTypeScopedPriorityRule
from typing import List


class EDDPriorityRule(JobTypeScopedPriorityRule):
    def assign_priority(self) -> List[int]:
        """
        각 JobTemplate 그룹 내에서 완료되지 않은 JobInstance 중 마감기한이 가장 빠른 인스턴스의 id 반환.
        모든 인스턴스가 완료되었다면 -1 반환.
        """
        result = []

        for job_group in self.scheduler.job_instances:
            # 완료되지 않은 인스턴스만 필터링
            unfinished_jobs = [
                job for job in job_group if not getattr(job, "completed", False)
            ]

            if not unfinished_jobs:
                result.append(-1)  # 모든 인스턴스 완료
                continue

            # deadline 기준으로 오름차순 정렬
            sorted_group = sorted(
                unfinished_jobs,
                key=lambda job: (
                    job.profit_fn.deadline if job.profit_fn else float("inf")
                ),
            )

            top_priority_instance = sorted_group[0]
            result.append(top_priority_instance.job_instance_id)

        return result
