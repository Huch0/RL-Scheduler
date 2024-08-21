from train.train_model import train_model
from train.test_model import test_model
from train.make_env import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor


# def evaluate_maskable_policy(model, env, n_eval_episodes=5, render=False):
#     episode_rewards = []
#     for _ in range(n_eval_episodes):
#         obs, _ = env.reset()
#         done = False
#         episode_reward = 0.0
#         while not done:
#             action_masks = env.action_masks()  # 마스킹된 행동 불러오기
#             action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
#             obs, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             episode_reward += reward
#             if done and render:
#                 env.render()
#         episode_rewards.append(episode_reward)
#     return np.mean(episode_rewards), np.std(episode_rewards)

if __name__ == "__main__":
    # learning_rates = [0.0003, 0.0001, 0.00001]
    # nsteps = [2048, 4096, 8192]
    # results = {}

    cost_list = [5, 1, 2, 10]
    profit_per_time = 10
    max_time = 50

    # ---------No Heatmap Test---------------------------------------------
    # env1_no_heatmap, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = False, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time, has_heatmap = False)
    # env2_no_heatmap, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = True, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time, has_heatmap = False)

    # model_no_heatmap = train_model(env = env1_no_heatmap, eval_env= env2_no_heatmap, env_name= "Single_Env4_no_heatmap_", version= "v1", total_steps= 1000000)
    # test_model(env=env2_no_heatmap, model=model_no_heatmap)

    # ---------Heatmap Test---------------------------------------------
    env_normal, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = False, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)
    env_test, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = True, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)
    env_tiny_normal, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = False, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time, sample_mode="tiny_normal")
    env_tiny_stairs, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = False, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time, sample_mode="tiny_stairs")
    
    env_normal_2, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [2] * 12, test_mode = False, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)
    env_normal_1_5, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1.5] * 12, test_mode = False, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)
    env_normal_0_5, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [0.5] * 12, test_mode = False, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)

    env_list = [lambda : env_normal, lambda: env_normal_2, lambda: env_normal_1_5, lambda : env_normal_0_5]
    vec_env = SubprocVecEnv(env_list)

    params = {
        "policy_kwargs": dict(
            net_arch=[256, 128, 64]
        ),
        "learning_rate": 0.00005,
        "gamma": 1.0,
    }

    model = train_model(env = vec_env, env_name= "Quad_Env4_all_normal", eval_env= env_normal, params=params,version= "v3", total_steps= 2000000, deterministic = False)
    test_model(env=env_test, model=model, deterministic=True, plot_policy=True)
    
    # for nstep in nsteps:
    #     print(f"Testing learning rate: {nstep}")
    #     eval_env = env_test
    #     eval_env = Monitor(eval_env)
    #     eval_env.reset()
    #     # 설정된 학습률로 모델 생성
    #     params = {
    #         "policy_kwargs": dict(
    #             net_arch=[256, 128, 64]
    #         ),
    #         "n_steps": nstep,
    #     }
    #     if nstep == 2048:
    #         model = MaskablePPO.load("./models/Env4/MP_Multi_Env4_binary_heatmap_v4", env = vec_env, policy_kwargs=params["policy_kwargs"])
    #     else:
    #         # 모델 학습
    #         model = MaskablePPO('MultiInputPolicy', vec_env, verbose=1, **params)
    #         log_path = "./logs/tmp/nsteps" + str(nstep)
    #         maskable_eval_callback = MaskableEvalCallback(eval_env, best_model_save_path=log_path,
    #                                               log_path=log_path, eval_freq=10000,
    #                                               deterministic=True, render=False)
    #         model.learn(total_timesteps=1500000, callback=maskable_eval_callback)
    #         model.save(f"./models/Env4/MP_Multi_Env4_binary_heatmap_v4_nstep_{nstep}")

    #     # 평가
    #     print("Evaluating model")
    #     mean_reward, std_reward = evaluate_maskable_policy(model, eval_env)

    #     # 결과 저장
    #     results[nstep] = (mean_reward, std_reward)
    #     print(f"Learning rate: {nstep}, Mean reward: {mean_reward}, Std: {std_reward}")

    # # 가장 좋은 학습률 찾기
    # best_nstep = max(results, key=lambda nstep: results[nstep][0])
    # print(f"Best nstep: {best_nstep} with mean reward: {results[best_nstep][0]}")

    # repeats = [
    #     [4, 3, 3, 5, 4, 2, 3, 2, 2, 3, 3, 4],
    #     [3, 3, 3, 3, 4, 2, 3, 2, 1, 3, 3, 2],
    #     [5, 1, 3, 2, 4, 4, 3, 3, 2, 1, 2, 3],
    #     [4, 4, 2, 2, 1, 1, 1, 4, 2, 2, 1, 3],
    #     [1, 2, 2, 3, 2, 1, 2, 3, 3, 3, 2, 2],
    #     [2, 2, 2, 1, 3, 2, 1, 3, 2, 3, 3, 3],
    #     [4, 1, 3, 2, 2, 2, 2, 3, 1, 3, 3, 1],
    #     [4, 4, 4, 2, 1, 4, 2, 4, 3, 3, 3, 3],
    #     [3, 4, 3, 3, 4, 1, 1, 3, 1, 4, 2, 2],
    #     [4, 4, 4, 3, 2, 4, 2, 3, 3, 2, 3, 3]
    # ]
    
    # model = MaskablePPO.load("./models/Env4/MP_Multi_Env4_binary_heatmap_v4", **params)


    # # for repeat in repeats:
        
    # #     env, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = repeat, repeat_stds = [1] * 12, test_mode = True, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)
    # #     test_model(env=env, model=model)
    
    # repeat = repeats[0]
    # env, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = repeat, repeat_stds = [1] * 12, test_mode = True, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)
    # test_model(env=env, model=model, deterministic=False)