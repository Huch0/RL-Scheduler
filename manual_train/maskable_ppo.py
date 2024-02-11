from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks

from scheduler_env.jssEnv import jssEnv

env = jssEnv()
# model = MaskablePPO("MultiInputPolicy", env, seed=32, verbose=1)
# model.learn(10000)

# evaluate_policy(model, env, n_eval_episodes=20,
#                 reward_threshold=90, warn=False)

# model.save("ppo_mask")
# del model  # remove to demonstrate saving and loading

model = MaskablePPO.load("ppo_mask")

obs, _ = env.reset()
done = False
episode_reward = 0
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode_reward += reward


print(f"Episode reward: {episode_reward}")
print(env.last_solution, env.last_time_step)
# fig = env.render()
# fig.show()
