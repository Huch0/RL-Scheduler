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


def make_env(
    scheduler: Scheduler,
    contract_generator: ContractGenerator | Type[ContractGenerator] | None = None,
    action_handler: str | Type[ActionHandler] | None = None,
    observation_handler: str | Type[ObservationHandler] | None = None,
    reward_handler: str | Type[RewardHandler] | None = None,
    info_handler: str | Type[InfoHandler] | None = None,
):
    # Resolve / instantiate gerator / handlers ------------------------------------
    # If caller passed an already‑constructed object, use it as‑is;
    # otherwise ask the registry for one.
    if not isinstance(contract_generator, ContractGenerator):
        contract_generator = get_contract_generator(contract_generator)

    if not isinstance(action_handler, ActionHandler):
        action_handler = get_action_handler(action_handler, scheduler)

    if not isinstance(observation_handler, ObservationHandler):
        observation_handler = get_observation_handler(observation_handler, scheduler)

    if not isinstance(reward_handler, RewardHandler):
        reward_handler = get_reward_handler(reward_handler, scheduler)

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
