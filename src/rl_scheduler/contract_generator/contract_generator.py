from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from rl_scheduler.scheduler.profit import ProfitFunction


class ContractGenerator(ABC):
    """
    Abstract base class for generating contracts.
    """

    def __init__(self, contract_path: Path):
        self.contract_path = contract_path

    @abstractmethod
    def load_profit_fn() -> List[List[ProfitFunction]]:
        """
        Load profit functions from a file.
        """
        pass

    @abstractmethod
    def load_repetition() -> List[int]:
        """
        Load repetition data from a file.
        """
        pass
