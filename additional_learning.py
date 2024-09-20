from train.train_model import train_model
from train.test_model import test_model
from train.make_env import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
# SB3 check_env import
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
import torch.nn as nn
import warnings
import math

warnings.filterwarnings("ignore")

cost_list = [5, 1, 2, 10]
profit_per_time = 10
max_time = 50

env4, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = False, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)

model_path = "MP_Single_Env4_gamma_1_obs_v4_clip_1_lr_custom_expv1"
model = MaskablePPO.load(model_path)

model.set_env(env4)
model.learning_rate = 0.00003

log_path = "logs/tmp/Env4"
new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])

maskable_eval_callback = MaskableEvalCallback(env4, best_model_save_path=log_path,
                                                  log_path=log_path, eval_freq=40960,
                                                  deterministic=False, render=False)

# 추가 학습 (300만 스텝)
model.learn(total_timesteps=3000000, callback = maskable_eval_callback)
model.save(model_path + "_8000000")