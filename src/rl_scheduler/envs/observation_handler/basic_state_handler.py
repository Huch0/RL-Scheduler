import numpy as np
from gymnasium import spaces
from .observation_handler import ObservationHandler


class BasicStateHandler(ObservationHandler):
    def __init__(self, scheduler):
        super().__init__(scheduler)

    def create_observation_space(self) -> spaces.Space:
        """
        Define the observation space for a dictionary-based observation.

        Note: This is a high-level placeholder implementation.
        The observation space should be redefined in detail later.
        """
        return spaces.Dict(
            {
                "machine_states": spaces.Box(
                    low=0, high=100, shape=(10,), dtype=np.int8
                ),
                "job_states": spaces.Box(low=0, high=100, shape=(10,), dtype=np.int8),
            }
        )

    def get_observation(self) -> dict:
        """
        Generate a dictionary-based observation.

        Note: This is a high-level placeholder implementation.
        The observation logic should be reimplemented with more detailed logic later.
        """
        return {
            "machine_states": [
                len(machine.assigned_operations)
                for machine in self.scheduler.machine_instances
            ],
            "job_states": [
                len(
                    self.scheduler.job_instances[t_id][i_id].operation_instance_sequence
                )
                for i_id in range(len(self.scheduler.job_instances[0]))
                for t_id in range(len(self.scheduler.job_instances))
            ],
        }
