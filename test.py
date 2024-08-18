from train.train_model import train_model
from train.test_model import test_model
from train.make_env import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO

if __name__ == "__main__":

    cost_list = [5, 1, 2, 10]
    profit_per_time = 10
    max_time = 50

    # ---------No Heatmap Test---------------------------------------------
    # env1_no_heatmap, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = False, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time, has_heatmap = False)
    # env2_no_heatmap, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = True, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time, has_heatmap = False)

    # model_no_heatmap = train_model(env = env1_no_heatmap, eval_env= env2_no_heatmap, env_name= "Single_Env4_no_heatmap_", version= "v1", total_steps= 1000000)
    # test_model(env=env2_no_heatmap, model=model_no_heatmap)

    # ---------Heatmap Test---------------------------------------------
    env1, env_name = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = False, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)
    env2, env_name = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = True, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)
    # Define your environments (env1 and env2 should be your custom environments)
    env1_lambda = lambda: env1
    env2_lambda = lambda: env2

    # List of environments
    env_list = [env1_lambda, env2_lambda]
    vec_env = SubprocVecEnv(env_list)

    # env1.is_image()
    # model = train_model(env = vec_env, eval_env= env1, env_name= "Multi_Env4_binary_heatmap_", version="v4", total_steps=1500000, net_arch=[256,128,64])
    # test_model(env=env2, model=model)

    policy_kwargs = dict(
        net_arch=[256, 128, 64]
    )

    repeats = [
        [4, 3, 3, 5, 4, 2, 3, 2, 2, 3, 3, 4],
        [3, 3, 3, 3, 4, 2, 3, 2, 1, 3, 3, 2],
        [5, 1, 3, 2, 4, 4, 3, 3, 2, 1, 2, 3],
        [4, 4, 2, 2, 1, 1, 1, 4, 2, 2, 1, 3],
        [1, 2, 2, 3, 2, 1, 2, 3, 3, 3, 2, 2],
        [2, 2, 2, 1, 3, 2, 1, 3, 2, 3, 3, 3],
        [4, 1, 3, 2, 2, 2, 2, 3, 1, 3, 3, 1],
        [4, 4, 4, 2, 1, 4, 2, 4, 3, 3, 3, 3],
        [3, 4, 3, 3, 4, 1, 1, 3, 1, 4, 2, 2],
        [4, 4, 4, 3, 2, 4, 2, 3, 3, 2, 3, 3]
    ]
    
    model = MaskablePPO.load("./models/Env4/MP_Multi_Env4_binary_heatmap_v4", env = vec_env, policy_kwargs=policy_kwargs)

    for repeat in repeats:
        env, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = repeat, repeat_stds = [1] * 12, test_mode = True, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)
        test_model(env=env, model=model)
    
    # env2, env_name = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = [3] * 12, repeat_stds = [1] * 12, test_mode = True, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)

    