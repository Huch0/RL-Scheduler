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

# class TrainEveryNEpisodesCallback(BaseCallback):
#     def __init__(self, model, n_episodes=20, verbose=0):
#         super(TrainEveryNEpisodesCallback, self).__init__(verbose)
#         self.model = model
#         self.n_episodes = n_episodes
#         self.episode_count = 0

#     def _on_step(self) -> bool:
#         # Check if the current episode is done
#         if self.locals['dones'][0]:
#             # Increment the episode count
#             self.episode_count += 1

#             # When the episode count reaches the desired number, trigger training
#             if self.episode_count >= self.n_episodes:
#                 print(f"Training after {self.episode_count} episodes.")
#                 self.model.train()  # Trigger training
#                 self.episode_count = 0  # Reset the episode count

#         return True


def train_model(env, env_name, version = "v1"):
    log_path = "./logs/tmp/" + env_name
    # set up logger
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    # Create the evaluation environment
    eval_env = env

    # Create the MaskablePPO model first
    model = MaskablePPO('MultiInputPolicy', env, verbose=1,
                        n_steps=4096)
    model.set_logger(new_logger)

    # Create the MaskableEvalCallback
    maskable_eval_callback = MaskableEvalCallback(eval_env, best_model_save_path=log_path,
                                                  log_path=log_path, eval_freq=100000,
                                                  deterministic=True, render=False)
    
    # Now that the model is created, pass it to the TrainEveryNEpisodesCallback
    # train_every_n_episodes_callback = TrainEveryNEpisodesCallback(model, n_episodes=20, verbose=1)

    # Combine the callbacks into a CallbackList
    #callback = CallbackList([maskable_eval_callback, train_every_n_episodes_callback])

    # Start the learning process
    model.learn(1000000, callback=maskable_eval_callback)

    # Save the trained model
    model.save("MP_" + env_name + version)

    return model