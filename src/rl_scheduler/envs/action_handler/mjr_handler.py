from .action_handler import ActionHandler
from gymnasium import spaces
from rl_scheduler.scheduler.scheduler import Scheduler
import numpy as np


class MJRHandler(ActionHandler):
    def __init__(self, scheduler: Scheduler, max_repetition: int = 5):
        """
        Initialize the MJRHandler with a Scheduler instance and max_repetition.

        Args:
            scheduler (Scheduler): The Scheduler object to base the action space on.
            max_repetition (int): The maximum number of repetitions.
        """
        self.max_repetition = max_repetition
        super().__init__(scheduler)

    def create_action_space(self) -> spaces.Space:
        """
        Create a gymnasium action space based on the Scheduler's configuration.

        Returns:
            spaces.Space: A gymnasium space representing the action space.
        """
        return spaces.MultiDiscrete(
            [
                len(self.scheduler.machine_templates),  # Number of machines (M)
                len(self.scheduler.job_templates),  # Number of jobs (J)
                self.max_repetition,  # Max repetitions (R)
            ]
        )

    def convert_action(self, action) -> tuple:
        """
        Convert an action from the action space into M x J x R form.

        Args:
            action: The action to convert.

        Returns:
            tuple: A tuple (M, J, R) representing the machine, job, and repetition.
        """
        return tuple(action)

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
        space = self.action_space

        # For this generic handler we assume the action is a tuple
        # (machine_idx, job_idx, repetition_idx).

        machine_dim, job_dim, rep_dim = (int(n) for n in space.nvec)
        mask = np.zeros((machine_dim, job_dim, rep_dim), dtype=bool)

        scheduler = self.scheduler
        machines = scheduler.machine_instances
        jobs = scheduler.job_instances

        for m_idx in range(machine_dim):
            machine = machines[m_idx]
            for j_idx in range(job_dim):
                for r_idx in range(rep_dim):
                    # Guard against missing repetitions
                    if j_idx >= len(jobs) or r_idx >= len(jobs[j_idx]):
                        continue

                    job_instance = jobs[j_idx][r_idx]

                    # Skip if job already completed
                    if job_instance.completed:
                        continue

                    # Next operation instance to process
                    try:
                        op_instance = scheduler.find_op_instance_by_action(j_idx, r_idx)
                    except AttributeError:
                        # Fallback: first uncompleted op
                        op_instance = next(
                            (
                                op
                                for op in job_instance.operation_instance_sequence
                                if not op.completed
                            ),
                            None,
                        )
                    if op_instance is None:
                        continue

                    # Check the scheduler constraint
                    mask[m_idx, j_idx, r_idx] = bool(
                        scheduler.check_constraint(machine, op_instance)
                    )

        return mask

    def sample_valid_action(self, rng: "np.random.Generator"):
        """
        Sample a random valid action using the mask from
        :py:meth:`compute_action_mask`.

        Parameters
        ----------
        rng : numpy.random.Generator | None, optional
            Pseudo‑random number generator to use for sampling.  If *None*,
            the method will try ``self.scheduler.np_random`` and finally fall
            back to ``np.random.default_rng()``.  Supplying the RNG from the
            Streamlit layer ensures reproducible behaviour that matches the
            user’s seed.

        Returns
        -------
        Any
            An action compatible with this handler's ``convert_action``.
        Raises
        ------
        RuntimeError
            If no valid actions are available.
        """
        mask = self.compute_action_mask()
        flat_valid_indices = np.flatnonzero(mask)

        if flat_valid_indices.size == 0:
            raise RuntimeError("No valid actions available.")

        chosen_flat = rng.choice(flat_valid_indices)
        # For MultiDiscrete spaces, unravel into (machine, job, repetition)
        dims = tuple(int(n) for n in self.action_space.nvec)
        chosen_idx = np.unravel_index(chosen_flat, dims)

        # Ensure we return a plain tuple of Python ints
        return tuple(int(x) for x in chosen_idx)
