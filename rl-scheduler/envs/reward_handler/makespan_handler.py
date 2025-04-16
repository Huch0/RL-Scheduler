from scheduler.Scheduler import Scheduler
from .reward_handler import RewardHandler


class MakespanHandler(RewardHandler):
    def __init__(self, scheduler: Scheduler, **kwargs):
        super().__init__(scheduler, **kwargs)

    def get_intermediate_reward(self) -> float:
        return 0.0

    def get_terminal_reward(self) -> float:
        # Calculate the makespan as the maximum end time of all operations
        makespan = max(
            machine.last_assigned_end_time
            for machine in self.scheduler.machine_instances
        )
        # Reward is the negative of the makespan
        return -makespan
