from rl_scheduler.scheduler.scheduler import Scheduler
from .info_handler import InfoHandler


class BasicInfoHandler(InfoHandler):
    def __init__(self, scheduler: Scheduler):
        """
        Initialize the BasicInfoHandler with a Scheduler instance.

        Args:
            scheduler (Scheduler): The Scheduler object to base the info computation on.
        """
        super().__init__(scheduler)

    def get_info(self) -> dict:
        """
        Compute and return an info dictionary based on the current state of the Scheduler.

        Returns:
            dict: A dictionary containing basic information about the Scheduler.
        """
        return {
            "num_machines": len(self.scheduler.machine_instances),
            "num_jobs": len(self.scheduler.job_instances),
            "all_jobs_scheduled": self.scheduler.is_all_job_instances_scheduled(),
        }
