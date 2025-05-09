import gymnasium as gym
from pathlib import Path
from rl_scheduler.scheduler.scheduler import Scheduler
from rl_scheduler.contract_generator import ContractGenerator
from .action_handler import ActionHandler
from .observation_handler import ObservationHandler
from .reward_handler import RewardHandler
from .info_handler import InfoHandler


class RJSPEnv(gym.Env):
    def __init__(
        self,
        scheduler: Scheduler,
        contract_generator: ContractGenerator,
        action_handler: ActionHandler,
        observation_handler: ObservationHandler,
        reward_handler: RewardHandler,
        info_handler: InfoHandler,
    ):
        super().__init__()

        # Initialize Scheduler
        self.scheduler = scheduler

        self.contract_generator = contract_generator

        # Define action space
        self.action_handler = action_handler
        self.action_space = self.action_handler.action_space

        # Define observation space
        self.observation_handler = observation_handler
        self.observation_space = observation_handler.observation_space

        # Define reward handler
        self.reward_handler = reward_handler

        # Define info handler
        self.info_handler = info_handler

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Extract contract_path from options if provided
        contract_path = None
        if options and "contract_path" in options:
            contract_path = Path(options["contract_path"])
        else:
            raise ValueError("contract_path must be provided in options.")

        # Load repetitions and profit functions
        repetitions = self.contract_generator.load_repetition(contract_path)
        profit_functions = self.contract_generator.load_profit_fn(contract_path)

        # Reset the scheduler
        self.scheduler.reset(repetitions=repetitions, profit_functions=profit_functions)

        # Return the initial obs, info
        return self.observation_handler.get_observation(), self.info_handler.get_info()

    def step(self, action):
        converted_action = self.action_handler.convert_action(action)
        chosen_machine_id, chosen_job_id, chosen_repetition = converted_action

        # Execute a scheduling step
        try:
            self.scheduler.step(chosen_machine_id, chosen_job_id, chosen_repetition)
        except ValueError as e:
            # Invalid action: keep env state unchanged and notify caller.
            info = self.info_handler.get_info()
            info.update(
                {
                    "invalid_action": True,
                    "error": str(e),
                }
            )
            # Small negative reward to discourage invalid moves; no termination.
            return (
                self.observation_handler.get_observation(),
                -0.1,
                False,  # terminated
                False,  # truncated
                info,
            )

        # Get the current observation
        obs = self.observation_handler.get_observation()

        # Check if all job instances are scheduled
        terminated = self.scheduler.is_all_job_instances_scheduled()
        truncated = False

        # Get the reward
        if terminated:
            reward = self.reward_handler.get_terminal_reward()
        else:
            reward = self.reward_handler.get_intermediate_reward()

        # Get additional info
        info = self.info_handler.get_info()

        # Return observation, reward, done, and additional info
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Optional: Implement visualization of the environment
        pass

    def close(self):
        # Optional: Clean up resources
        pass
