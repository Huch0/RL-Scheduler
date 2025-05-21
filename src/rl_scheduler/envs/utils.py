from pathlib import Path

from .rjsp_env import RJSPEnv
from .registry import (
    get_contract_generator,
    get_action_handler,
    get_observation_handler,
    get_reward_handler,
    get_info_handler,
)
from rl_scheduler.contract_generator import ContractGenerator
from .action_handler import ActionHandler
from .observation_handler import ObservationHandler
from .reward_handler import RewardHandler
from .info_handler import InfoHandler
from rl_scheduler.scheduler import Scheduler
from typing import Type
from rl_scheduler.envs.action_handler import MJHandler


def make_env(
    scheduler: dict | Scheduler,
    contract_generator: ContractGenerator | Type[ContractGenerator] | None = None,
    contract_path: Path | str | None = None,
    action_handler: str | Type[ActionHandler] | None = None,
    action_handler_kwargs: dict | None = None,
    observation_handler: str | Type[ObservationHandler] | None = None,
    observation_handler_kwargs: dict | None = None,
    reward_handler: str | Type[RewardHandler] | None = None,
    reward_handler_kwargs: dict | None = None,
    info_handler: str | Type[InfoHandler] | None = None,
):
    # Instantiate scheduler if passed as a dict
    if isinstance(scheduler, dict):
        scheduler = Scheduler(
            machine_config_path=scheduler["machine_config_path"],
            job_config_path=scheduler["job_config_path"],
            operation_config_path=scheduler["operation_config_path"],
        )
    # Check if scheduler is an instance of Scheduler
    if not isinstance(scheduler, Scheduler):
        raise TypeError("Scheduler must be a dict or an instance of Scheduler.")

    # Resolve / instantiate generator / handlers ------------------------------------
    # If caller passed an already‑constructed object, use it as‑is;
    # otherwise ask the registry for one.
    if not isinstance(contract_generator, ContractGenerator):
        contract_generator = get_contract_generator(
            contract_generator, Path(contract_path)
        )

    if not isinstance(action_handler, ActionHandler):
        action_handler = get_action_handler(
            action_handler, scheduler, **(action_handler_kwargs or {})
        )

    if not isinstance(observation_handler, ObservationHandler):
        # Special case
        if observation_handler == "mlp" or observation_handler == "cnn":
            # MLPHandler requires a MJHandler
            if not isinstance(action_handler, MJHandler):
                raise ValueError(
                    "MLPHandler requires MJHandler as action handler. "
                    "Please set action_handler='mj' when using MLPHandler."
                )
            observation_handler = get_observation_handler(
                observation_handler,
                scheduler,
                mj_action_handler=action_handler,
                **(observation_handler_kwargs or {}),
            )
        else:
            observation_handler = get_observation_handler(
                observation_handler, scheduler, **(observation_handler_kwargs or {})
            )

    if not isinstance(reward_handler, RewardHandler):
        reward_handler = get_reward_handler(
            reward_handler, scheduler, **(reward_handler_kwargs or {})
        )

    if not isinstance(info_handler, InfoHandler):
        info_handler = get_info_handler(info_handler, scheduler)

    # -------------------------------------------------------------------
    return RJSPEnv(
        scheduler=scheduler,
        contract_generator=contract_generator,
        action_handler=action_handler,
        observation_handler=observation_handler,
        reward_handler=reward_handler,
        info_handler=info_handler,
    )
