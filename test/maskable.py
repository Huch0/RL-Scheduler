import gymnasium as gym
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import EvalCallback
from scheduler_env.customEnv import SchedulingEnv

# env = SchedulingEnv()
# eval_env = SchedulingEnv()

# model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env, verbose=1)

# model_path = "./logs/"
# # Use deterministic actions for evaluation
# eval_callback = EvalCallback(eval_env, best_model_save_path=model_path,
#                              log_path="./logs/", eval_freq=100000,
#                              deterministic=True, render=False)

# model.learn(total_timesteps=1000000, callback=eval_callback)

# print("Best model saved at:", model_path)

# del model  # remove to demonstrate saving and loading

model = MaskablePPO.load("./logs/best_model.zip")

vec_env = SchedulingEnv()
evaluate_policy(model, vec_env, n_eval_episodes=20,
                reward_threshold=30, warn=False)

reward_sum = 0
obs, _ = vec_env.reset()
while True:
    action_masks = get_action_masks(vec_env)
    action, _states = model.predict(
        obs, deterministic=True, action_masks=action_masks)
    obs, reward, terminated, truncated, info = vec_env.step(action)

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
