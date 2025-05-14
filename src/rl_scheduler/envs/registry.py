from __future__ import annotations
from pathlib import Path
from typing import Mapping, Type

from rl_scheduler.contract_generator import ContractGenerator, DeterministicGenerator

from .action_handler import MJHandler, MJRHandler, ActionHandler
from .observation_handler import MLPHandler, BasicStateHandler, ObservationHandler
from .reward_handler import ProfitCostHandler, MakespanHandler, RewardHandler
from .info_handler import InfoHandler, BasicInfoHandler
from rl_scheduler.scheduler import Scheduler


_CONTRACT_GENERATORS: Mapping[str, Type[ContractGenerator]] = {
    "deterministic": DeterministicGenerator,
    "stochastic": StochasticGenerator,
}

_ACTION_HANDLERS: Mapping[str, Type[ActionHandler]] = {
    "mjr": MJRHandler,
    "mj": MJHandler,
}

_OBSERVATION_HANDLERS: Mapping[str, Type[ObservationHandler]] = {
    "basic": BasicStateHandler,
    "mlp": MLPHandler,
}

_REWARD_HANDLERS: Mapping[str, Type[RewardHandler]] = {
    "makespan": MakespanHandler,
    "profit_cost": ProfitCostHandler,
}

_INFO_HANDLERS: Mapping[str, Type[InfoHandler]] = {
    "basic": BasicInfoHandler,
}


def get_contract_generator(
    name: str | None,
    contract_path: str | Path,
) -> ContractGenerator:
    if name is None:
        return DeterministicGenerator(contract_path)
    cls = _CONTRACT_GENERATORS.get(name.lower())
    if cls is None:
        raise KeyError(f"Unknown contract generator: {name!r}")
    return cls(contract_path)


def get_action_handler(
    name: str | None, scheduler: Scheduler, **kwargs
) -> ActionHandler:
    if name is None:
        return MJRHandler(scheduler)
    cls = _ACTION_HANDLERS.get(name.lower())
    if cls is None:
        raise KeyError(f"Unknown action handler: {name!r}")
    return cls(scheduler, **kwargs)


def get_observation_handler(
    name: str | None, scheduler: Scheduler, **kwargs
) -> ObservationHandler:
    if name is None:
        return BasicStateHandler(scheduler)
    cls = _OBSERVATION_HANDLERS.get(name.lower())
    if cls is None:
        raise KeyError(f"Unknown observation handler: {name!r}")
    return cls(scheduler, **kwargs)


def get_reward_handler(
    name: str | None, scheduler: Scheduler, **kwargs
) -> RewardHandler:
    if name is None:
        return MakespanHandler(scheduler)
    cls = _REWARD_HANDLERS.get(name.lower())
    if cls is None:
        raise KeyError(f"Unknown reward handler: {name!r}")
    return cls(scheduler, **kwargs)


def get_info_handler(name: str | None, scheduler: Scheduler) -> InfoHandler:
    if name is None:
        return BasicInfoHandler(scheduler)
    cls = _INFO_HANDLERS.get(name.lower())
    if cls is None:
        raise KeyError(f"Unknown info handler: {name!r}")
    return cls(scheduler)
