from gymnasium.envs.registration import register

register(
    id="RJSPEnv-v0",
    entry_point="rl_scheduler.envs:make_env",
)
