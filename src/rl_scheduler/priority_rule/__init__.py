from .priority_rule import PriorityRule
from .job_type_scope.job_type_scope_priority_rule import JobTypeScopedPriorityRule
from .job_type_scope.edd_priority_rule import EDDPriorityRule
from .job_type_scope.etd_priority_rule import ETDPriorityRule
from .global_scope.global_scope_priority_rule import GlobalScopedPriorityRule
from rl_scheduler.scheduler import Scheduler

#
# Map IDs to JobTypeScopedPriorityRule subclasses
_JOB_TYPE_RULES: dict[str, type[JobTypeScopedPriorityRule]] = {
    "edd": EDDPriorityRule,
    "etd": ETDPriorityRule,
}

# Map IDs to GlobalScopedPriorityRule subclasses
_GLOBAL_RULES: dict[str, type[GlobalScopedPriorityRule]] = {
    "global": GlobalScopedPriorityRule,
}

__all__ = [
    "PriorityRule",
    "JobTypeScopedPriorityRule",
    "GlobalScopedPriorityRule",
    "EDDPriorityRule",
    "ETDPriorityRule",
    "get_job_type_rule",
    "get_global_rule",
]


def get_job_type_rule(rule_id: str, scheduler: Scheduler) -> JobTypeScopedPriorityRule:
    """
    Return a JobTypeScopedPriorityRule based on `rule_id`.
    Valid keys: {edd, etd}.
    """
    try:
        cls = _JOB_TYPE_RULES[rule_id]
    except KeyError:
        valid = ", ".join(sorted(_JOB_TYPE_RULES.keys()))
        raise ValueError(f"Unknown job-type rule '{rule_id}'. Valid: {valid}")
    return cls(scheduler)


def get_global_rule(scheduler: Scheduler) -> GlobalScopedPriorityRule:
    """
    Return a GlobalScopedPriorityRule instance.
    """
    # Only one global rule available
    cls = GlobalScopedPriorityRule
    return cls(scheduler)
