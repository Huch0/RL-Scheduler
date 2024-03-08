import gymnasium as gym
from scheduler_env.customEnv import SchedulingEnv

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO.load(
    "../rl_zoo3/logs/ppo/scheduler-customEnv_2/best_model.zip")
env = SchedulingEnv()

# mean_reward, std_reward = evaluate_policy(
#     model, env, n_eval_episodes=10)


vec_env = env
obs, _ = vec_env.reset()
reward_sum = 0

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = vec_env.step(action)
    # print(info['finish_time'])
    # vec_env.render()
    reward_sum += reward
    if terminated or truncated:
        print("Goal reached!", "final score=", reward / 37 * 100)
        print('finish_time', info['finish_time'])
        print('order_density', info['order_density'])
        print('order_score', info['order_score'])
        print('resource_operation_rate', info['resource_operation_rate'])
        print('resource_score', info['resource_score'])
        vec_env.render("seaborn")
        break
