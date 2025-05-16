import gymnasium as gym
from pathlib import Path
import pickle
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

    @property
    def timestep(self):
        """Get the current timestep."""
        return self.scheduler.timestep

    def action_masks(self) -> list[bool]:
        return self.action_handler.compute_action_mask().tolist()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Load repetitions and profit functions
        repetitions = self.contract_generator.load_repetition()
        profit_functions = self.contract_generator.load_profit_fn()

        # Reset the scheduler
        self.scheduler.reset(repetitions=repetitions, profit_functions=profit_functions)

        # Update static observation features
        self.observation_handler.update_static_observation_features()

        # Return the initial obs, info
        return self.observation_handler.get_observation(), self.info_handler.get_info()

    def step(self, action):
        # Execute a scheduling step
        try:
            converted_action = self.action_handler.convert_action(action)
            chosen_machine_id, chosen_job_id, chosen_repetition = converted_action
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

    def save(self, dir_path: str | Path):
        """
        Save the entire environment state to a single file for later restoration.
        """

        # Ensure the target directory exists
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save environment to env.pkl inside the directory
        file_path = dir_path / "env.pkl"
        with file_path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "RJSPEnv":
        """
        Load an environment saved via `save()` from the given file.
        """
        import pickle
        from pathlib import Path

        path = Path(path)
        with path.open("rb") as f:
            env = pickle.load(f)
        return env
