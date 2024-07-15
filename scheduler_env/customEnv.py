import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from stable_baselines3.common.env_checker import check_env
# from stable_baselines3 import A2C, PPO, DQN
# from sb3_contrib import MaskablePPO
# from sb3_contrib.common.wrappers import ActionMasker
from scheduler_env.customScheduler import customScheduler

class SchedulingEnv(gym.Env):
    def _load_machines(self, file_path):
        machines = []

        with open(file_path, 'r') as file:
            data = json.load(file)

        for machine_data in data["machines"]:
            machine = {}
            machine['name'] = machine_data["name"]
            machine['ability'] = machine_data["type"].split(', ')
            machines.append(machine)

        return machines

    def _load_jobs(self, file):
        # Just in case we are reloading operations

        jobs = []  # 리턴할 용도
        jobs_new_version = []  # 파일 읽고 저장할 때 쓰는 용도
        f = open(file)

        # returns JSON object as  a dictionary
        data = json.load(f)
        f.close()
        jobs_new_version = data['jobs']

        for job in jobs_new_version:
            job_info = {}
            # Initial index of steps within job
            job_info['name'] = job['name']
            job_info['color'] = job['color']
            # changed : add deadline
            job_info['deadline'] = job['deadline']
            earliestStart = job['earliest_start']

            operations = []
            for operation in job['operations']:
                predecessor = operation['predecessor']
                operation_info = {}
                # Sequence is the scheduling job, the series of which defines a State or Node.
                operation_info['sequence'] = None
                operation_info['index'] = operation['index']
                operation_info['type'] = operation['type']
                if predecessor is None:
                    operation_info['predecessor'] = None
                    operation_info['earliest_start'] = earliestStart
                else:
                    operation_info['predecessor'] = predecessor
                    operation_info['earliest_start'] = None
                operation_info['duration'] = operation['duration']
                operation_info['start'] = None
                operation_info['finish'] = None

                operations.append(operation_info)

            job_info['operations'] = operations
            jobs.append(job_info)

        return jobs

    def __init__(self, machine_config_path="instances/Machines/v0-8.json", job_config_path="instances/Jobs/v0-12-deadline.json", render_mode="seaborn", weight_final_time=80, weight_job_deadline=20, weight_op_rate=0, target_time = 1000):
        super(SchedulingEnv, self).__init__()
        self.weight_final_time = weight_final_time
        self.weight_job_deadline = weight_job_deadline
        self.weight_op_rate = weight_op_rate
        self.target_time = target_time

        self.job_config = self._load_jobs(job_config_path)
        self.machine_config = self._load_machines(machine_config_path)

        self.custom_scheduler = customScheduler(jobs = self.job_config, machines= self.machine_config)
        
        self.len_machines = len(self.machine_config)
        self.len_jobs = len(self.job_config)

        self.num_steps = 0

        self.action_space = spaces.Discrete(self.len_machines * self.len_jobs)
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(low=0, high=1, shape=(self.len_machines * self.len_jobs, ), dtype=np.int8),
            "job_details": spaces.Box(low=-1, high=25, shape=(self.len_jobs, 4, 2), dtype=np.int8),
            'job_density': spaces.Box(low=0, high=1, shape=(self.len_jobs, ), dtype=np.float32),
            'machine_operation_rate': spaces.Box(low=0, high=1, shape=(self.len_machines, ), dtype=np.float32),
            "num_operation_per_machine": spaces.Box(low=0, high=100, shape=(self.len_machines, ), dtype=np.int64),
            "machine_types": spaces.Box(low=0, high=1, shape=(self.len_machines, 25), dtype=np.int8),
            "operation_schedules": spaces.Box(low=0, high=1, shape=(self.len_machines, 50), dtype=np.int8)
        })

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.custom_scheduler.reset(seed=seed, options=options)
        self.num_steps = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Map the action to the corresponding machine and job
        selected_machine_id = action // self.len_jobs
        selected_job_id = action % self.len_jobs
        action = [selected_machine_id, selected_job_id]

        # error_action이 아니라면 step의 수를 증가시킨다
        self.num_steps += 1
        reward = 0

        self._update_legal_actions()

        if self._is_legal(action):
            self._update_state(action)
            reward = self._calculate_step_reward()
        else:  # Illegal action
            reward = -0.5

        terminated = self._is_done()
        if terminated:
            reward += self._calculate_final_reward()
            
        truncated = bool(self.num_steps == 10000)
        if truncated:
            reward = -100

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_legal_actions(self):
        self.custom_scheduler.update_legal_actions()

    def _is_legal(self, action):
        return self.custom_scheduler.is_legal(action)

    def _is_done(self):
        return self.custom_scheduler.is_done()

    def _update_state(self, action):
        self.custom_scheduler.update_state(action)

    def _get_observation(self):
        return self.custom_scheduler.get_observation()
    
    def _get_info(self):
        info = self.custom_scheduler.get_info()
        info['num_steps'] = self.num_steps
        return info

    def _calculate_final_reward(self):
        return self.custom_scheduler.calculate_final_reward(weight_final_time = self.weight_final_time, weight_job_deadline = self.weight_job_deadline, weight_op_rate = self.weight_op_rate, target_time = self.target_time)
    
    def _calculate_step_reward(self):
        return self.custom_scheduler.calculate_step_reward()

    def render(self, mode="seaborn"):
        self.custom_scheduler.render(mode = mode, num_steps = self.num_steps)

if __name__ == "__main__":
    env = SchedulingEnv(machines="instances/Machines/v0-8.json",
                        jobs="instances/Jobs/v0-12-deadline.json")

    check_env(env)

    step = 0
    obs, _ = env.reset()

    while True:
        step += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("Goal reached!", "reward=", reward)
            env.render()
            break
