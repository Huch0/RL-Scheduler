from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from rl_scheduler.scheduler.profit import ProfitFunction


class ContractGenerator(ABC):
    """
    Abstract base class for generating contracts.
    """

    @staticmethod
    @abstractmethod
    def load_profit_fn(file_path: Path) -> List[List[ProfitFunction]]:
        """
        Load profit functions from a file.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_repetition(file_path: Path) -> List[int]:
        """
        Load repetition data from a file.
        """
        pass
