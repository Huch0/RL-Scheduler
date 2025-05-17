from rl_scheduler.scheduler import Scheduler
from .reward_handler import RewardHandler
from rl_scheduler.envs.cost_functions import compute_costs


class ProfitCostHandler(RewardHandler):
    def __init__(self, scheduler: Scheduler, weights: dict[str, float] = None):
        super().__init__(scheduler)
        # Set default weights if not provided
        if weights is None:
            self.weights = {
                "C_max": 10.0,
                "C_mup": 2.0,
                "C_mid": 1.0,
            }
        else:
            self.weights = weights

    def get_intermediate_reward(self) -> float:
        return 0.0

    def get_terminal_reward(self) -> float:
        # P: total profit of all completed job instances
        P = sum(
            max(job_instance.get_profit(), 0)  # only count positive profit
            for job_type in self.scheduler.job_instances
            for job_instance in job_type
        )

        # C: weighted sum of each cost category
        # C_max: Total Makespan Cost
        # C_mup: Total Machine Uptime Cost
        # C_mid: Total Machine Idle Cost

        costs = compute_costs(self.scheduler)

        # Calculate the total cost
        # C = W_max * C_max + W_mup * C_mup + W_mid * C_mid
        C = (
            self.weights["C_max"] * costs["C_max"]
            + self.weights["C_mup"] * costs["C_mup"]
            + self.weights["C_mid"] * costs["C_mid"]
        )

        # R = (P - C) / P
        # Avoid division by zero
        if P == 0:
            return 0.0
        else:
            return (P - C) / P
