from .action_handler import ActionHandler
from gymnasium import spaces
from rl_scheduler.scheduler import Scheduler
from rl_scheduler.priority_rule import get_job_type_rule
import numpy as np


class MJHandler(ActionHandler):
    def __init__(self, scheduler: Scheduler, priority_rule_id: str = "etd"):
        """
        Initialize the MJRHandler with a Scheduler instance and max_repetition.

        Args:
            scheduler (Scheduler): The Scheduler object to base the action
            space on.
            max_repetition (int): The maximum number of repetitions.
        """
        self.M = len(scheduler.machine_templates)
        self.J = len(scheduler.job_templates)
        self.priority_rule = get_job_type_rule(priority_rule_id, scheduler)
        super().__init__(scheduler)

    def create_action_space(self) -> spaces.Space:
        """
        Create a gymnasium action space based on the Scheduler's configuration.

        Returns:
            spaces.Space: A gymnasium space representing the action space.
        """
        # Number of machines (M) * Number of jobs (J)
        return spaces.Discrete(self.M * self.J)

    def convert_action(self, action: int) -> tuple:
        """
        Convert an action from the action space into M x J x R form.

        Args:
            action: The action to convert, expected to be an integer.

        Returns:
            tuple: A tuple (M, J, R) representing the machine, job,
            and repetition.
        """

        # Expect `action` to be a tuple: (machine_index, job_template_index)
        # Convert to a flat index
        machine_idx = action // self.J
        job_idx = action % self.J

        # Check if machine and job indices are valid
        if machine_idx >= self.M:
            raise IndexError(f"Machine index {machine_idx} out of bounds")
        if job_idx >= self.J:
            raise IndexError(f"Job index {job_idx} out of bounds")

        # Determine repetition index via the priority rule
        # `assign_priority()` returns a list of chosen repetition per job
        # template
        priorities = self.priority_rule.assign_priority()
        repetition_idx = priorities[job_idx]
        if repetition_idx == -1:
            raise ValueError(
                f"All job instances are already assigned for job {job_idx}"
            )

        return machine_idx, job_idx, repetition_idx

    def compute_action_mask(self) -> "np.ndarray":
        """
        Generic validity mask for Discrete‑like action spaces.

        For `spaces.MultiDiscrete` or a `spaces.Tuple` of `spaces.Discrete`,
        we enumerate every possible coordinate and ask the scheduler whether
        the action is executable *right now*.

        Sub‑classes can override this method for efficiency, but the default
        should work for small/medium discrete spaces.

        Returns
        -------
        numpy.ndarray
            Boolean array with the same shape as the action space.
        """
        # Action: Number of machines (M) * Number of jobs (J)
        mask = np.zeros((self.M, self.J), dtype=np.int8)

        scheduler = self.scheduler
        machines = scheduler.machine_instances
        priorities = self.priority_rule.assign_priority()

        for j_idx in range(self.J):
            # Find the job instance with the highest priority
            r_idx = priorities[j_idx]

            # if all job instances are already assigned, skip
            if r_idx == -1:
                continue

            # Find the operation instance for the given job and repetition.
            op = scheduler.find_op_instance_by_action(j_idx, r_idx)

            for m_idx in range(self.M):
                machine = machines[m_idx]
                # Check if the machine can process the operation
                if scheduler.check_constraint(machine, op):
                    # Mark the action as valid
                    mask[m_idx, j_idx] = 1

        # Flatten the mask to match the action space
        return mask.flatten()
