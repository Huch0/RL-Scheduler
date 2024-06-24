import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import numpy as np
import json
import os

from scheduler_env.JSScheduler import JSScheduler


class JSSPEnv(gym.Env):
    def __init__(self, job_config_path, machine_config_path, max_job_repetition=3):
        super(JSSPEnv, self).__init__()

        with open(job_config_path, 'r') as f:
            self.job_config = json.load(f)
        with open(machine_config_path, 'r') as f:
            self.machine_config = json.load(f)

        self.n_jobs = self.job_config['n_jobs']
        self.n_machines = self.machine_config['n_machines']
        self.total_ops = sum([len(job['operation_queue']) for job in self.job_config['jobs']])
        self.max_n_operations = max([len(job['operation_queue']) for job in self.job_config['jobs']])
        self.max_job_repetition = max_job_repetition

        self.num_steps = 0

        self.JSScheduler = JSScheduler(self.job_config, self.machine_config, max_job_repetition=self.max_job_repetition)
        self.scheduler_state = self.JSScheduler.get_state()

        self.action_space = spaces.Discrete(self.n_machines * self.n_jobs)
        self.observation_space = spaces.Dict({
            'action_mask': spaces.Box(low=0, high=1, shape=(self.n_machines * self.n_jobs,), dtype=np.int8),
            'schedule_table': spaces.Box(low=-1, high=500, shape=(self.n_machines, self.total_ops * self.max_job_repetition, 6), dtype=np.int16),
            'job_buffer': spaces.Box(low=-1, high=500, shape=(self.n_jobs, self.max_job_repetition, self.max_n_operations, 6), dtype=np.int16),
            'machine_info': spaces.Box(low=-1, high=500, shape=(self.n_machines, 6), dtype=np.float32)
        })

    def reset(self, seed=0, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.num_steps = 0

        self.JSScheduler.reset()
        self.scheduler_state = self.JSScheduler.get_state()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.num_steps += 1

        machine_id = action // self.n_jobs
        job_id = action % self.n_jobs

        reward = 0

        try:
            self.JSScheduler.schedule_selected_job(machine_id, job_id)
        except Exception as e:
            # Invalid action
            # print(e)
            reward -= 100

        reward = self._get_step_reward()

        terminated = self.JSScheduler.is_done()

        if terminated:
            reward += self._get_final_reward()

        truncated = bool(self.num_steps >= 10000)

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_step_reward(self):
        mean_utilization = self.JSScheduler.get_mean_machine_utilization()

        return mean_utilization

    def _get_final_reward(self):
        makespan = self.JSScheduler.get_make_span()

        # Normalize the makespan
        return 1 / makespan

    def render(self, mode='human'):
        self.JSScheduler.show_gantt_chart()

    def _get_observation(self):
        return {
            'action_mask': np.logical_not(self.scheduler_state['valid_actions']).flatten(),
            'schedule_table': self.scheduler_state['schedule_table'],
            'job_buffer': self.scheduler_state['job_buffer'],
            'machine_info': self.scheduler_state['machine_info']
        }

    def _get_info(self):
        return self.JSScheduler.get_info()


if __name__ == "__main__":
    # ../instances
    instance_path = os.path.join(os.path.dirname(__file__), '../instances')
    env = JSSPEnv(job_config_path=os.path.join(instance_path, 'Jobs/v1-5.json', ),
                  machine_config_path=os.path.join(instance_path, 'Machines/v1-5.json'))
    check_env(env, warn=True)

    step = 0
    obs, _ = env.reset()
    done = False

    while not done:
        step += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            print("Goal reached!")
            print(info)
            env.render()

    env.close()
