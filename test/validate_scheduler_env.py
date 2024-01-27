# from scheduler_env import SchedulingEnv, load_orders
from stable_baselines3.common.env_checker import check_env
# import sys
# sys.path.append('../scheduler_env')

# env = SchedulingEnv(tasks=load_orders("./orders/orders-default.json"))
# # If the environment don't follow the interface, an error will be thrown
# check_env(env, warn=True)

# obs, _ = env.reset()

# print(env.observation_space)
# print(env.action_space)
# print(env.action_space.sample())

# step = 0
# while True:
#     step += 1
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
#     # env.render()
#     # print(action, reward, step)
#     # print("obs=", obs, "reward=", reward, "done=", done)
#     if done:
#         print("Goal reached!", "reward=", reward)
#         env.render()
#         break
