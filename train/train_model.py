from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList

from scheduler_env.customEnv_repeat import SchedulingEnv
import numpy as np

def make_env(
    num_machines,
    num_jobs,
    max_repeats,
    repeat_means,
    repeat_stds,
    test_mode=False,
    cost_list=None,
    profit_per_time=10,
    max_time=150
):
    if cost_list is None:
        cost_list = [5, 1, 2, 10]  # Default costs: [cost_deadline_per_time, cost_hole_per_time, cost_processing_per_time, cost_makespan_per_time]

    assert len(cost_list) == 4, "cost_list must contain exactly 4 elements."

    # 3. job_repeats_params 생성
    job_repeats_params = [(repeat_means[i], repeat_stds[i]) for i in range(len(repeat_means))]

    # 4. 환경 생성
    env = SchedulingEnv(
        machine_config_path= "instances/Machines/v0-" + str(num_machines) + ".json",  
        job_config_path= "instances/Jobs/v0-" + str(num_jobs) + "x" + str(max_repeats) + ".json", 
        job_repeats_params=job_repeats_params,
        render_mode="seaborn",
        cost_deadline_per_time=cost_list[0],
        cost_hole_per_time=cost_list[1],
        cost_processing_per_time=cost_list[2],
        cost_makespan_per_time=cost_list[3],
        profit_per_time=profit_per_time,
        target_time=None,
        test_mode=test_mode,
        max_time=max_time
    )

    # env 이름 생성
    env_name = f"Env_{str(num_machines)}_{str(num_jobs)}_{int(np.mean(repeat_means))}_{int(np.mean(repeat_stds))}_p{str(profit_per_time)}"

    return env, env_name

class LinearStdDecayScheduler:
    def __init__(self, initial_std, final_std, total_steps):
        self.initial_std = initial_std
        self.final_std = final_std
        self.total_steps = total_steps
        self.current_step = 0

    def get_current_std(self):
        # 선형적으로 표준편차를 감소시킴
        decay_rate = (self.initial_std - self.final_std) / self.total_steps
        current_std = max(self.final_std, self.initial_std - decay_rate * self.current_step)
        return current_std

    def step(self):
        # 스텝을 증가시키고 표준편차를 업데이트
        self.current_step += 1

class UpdateStdCallback(BaseCallback):
    def __init__(self, std_scheduler, verbose=0):
        super(UpdateStdCallback, self).__init__(verbose)
        self.std_scheduler = std_scheduler

    def _on_step(self) -> bool:
        # Update the standard deviation
        current_std = self.std_scheduler.get_current_std()
        # Optionally, log the current standard deviation
        self.logger.record("train/current_std", current_std)
        
        # Update the environment with the new std (if applicable)
        self.training_env.env_method("update_repeat_stds", current_std)

        # Step the scheduler
        self.std_scheduler.step()
        
        return True

if __name__ == "__main__":
    # Test the function
    env, env_name = make_env(
        num_machines=5,
        num_jobs_with_repeats=15,
        repeat_means=[3] * 5,
        repeat_stds=[1] * 5,
        test_mode=False,
        cost_list=[5, 1, 2, 10],
        profit_per_time=10
    )

    print(f"Environment Name: {env_name}")
    # Add additional checks or functionality as needed


def train_model(env, env_name, eval_env, version = "v1", total_steps = 1000000, net_arch = [256, 64]):
    log_path = "./logs/tmp/" + env_name
    # set up logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    # Create the evaluation environment

    policy_kwargs = dict(
        net_arch = net_arch
    )

    # 표준편차 스케줄러 초기화
    # std_scheduler = LinearStdDecayScheduler(initial_std, final_std, total_steps)

    # Create the MaskablePPO model first
    model = MaskablePPO('MultiInputPolicy', env, verbose=1,
                        policy_kwargs=policy_kwargs)
    model.set_logger(new_logger)

    # Create the MaskableEvalCallback
    maskable_eval_callback = MaskableEvalCallback(eval_env, best_model_save_path=log_path,
                                                  log_path=log_path, eval_freq=10000,
                                                  deterministic=True, render=False)
    
    # Create the custom callback for updating standard deviation
    # update_std_callback = UpdateStdCallback(std_scheduler)

    # callback = CallbackList([maskable_eval_callback, update_std_callback])

    # Start the learning process
    model.learn(total_steps, callback=maskable_eval_callback)

    # Save the trained model
    model.save("MP_" + env_name + version)

    return model