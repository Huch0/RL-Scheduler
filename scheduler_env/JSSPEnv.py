import gymnasium as gym
from gymnasium import spaces
import json
from .JSScheduler import JSScheduler


class JSSPEnv(gym.Env):
    def __init__(self, job_config_path, machine_config_path):
        super(JSSPEnv, self).__init__()

        with open(job_config_path, 'r') as f:
            self.job_config = json.load(f)
        with open(machine_config_path, 'r') as f:
            self.machine_config = json.load(f)

        self.n_jobs = self.job_config['n_jobs']
        self.n_machines = self.machine_config['n_machines']

        self.JSScheduler = JSScheduler(self.job_config, self.machine_config)
