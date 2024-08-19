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
from stable_baselines3.common.monitor import Monitor

from scheduler_env.customEnv_repeat import SchedulingEnv
import numpy as np

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


def train_model(env, env_name, eval_env, version = "v1", total_steps = 1000000, net_arch = [256, 64], algorithm = "MaskablePPO"):
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
    model_mp = MaskablePPO('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
    model_ppo = PPO('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
    model_dqn = DQN('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
    model_a2c = A2C('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs)

    models = [model_mp, model_ppo, model_dqn, model_a2c]

    model_index = 0
    model_name = ""
    if algorithm == "MaskablePPO":
        model_name = "MP_"
        model_index = 0
    elif algorithm == "PPO":
        model_name = "PPO_"
        model_index = 1
    elif algorithm == "DQN":
        model_name = "DQN_"
        model_index = 2
    elif algorithm == "A2C":
        model_name = "A2C_"
        model_index = 3

    model = models[model_index]

    model.set_logger(new_logger)

    # Create the MaskableEvalCallback
    # eval_env Monitor 감싸기
    eval_env = Monitor(eval_env)

    maskable_eval_callback = MaskableEvalCallback(eval_env, best_model_save_path=log_path,
                                                  log_path=log_path, eval_freq=10000,
                                                  deterministic=True, render=False)
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_path, log_path=log_path, 
                                eval_freq=10000, deterministic=True, render=False)
    
    # Create the custom callback for updating standard deviation
    # update_std_callback = UpdateStdCallback(std_scheduler)

    if model_index == 0:
        callback = maskable_eval_callback
    else:
        callback = eval_callback
    # callback = CallbackList([maskable_eval_callback, update_std_callback])

    # Start the learning process
    model.learn(total_steps, callback=callback)

    # Save the trained model
    model.save(model_name + env_name + version)

    return model