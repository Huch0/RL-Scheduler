from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import List
from rl_scheduler.scheduler.profit import ProfitFunction


class ContractGenerator(ABC):
    """
    Abstract base class for generating contracts.
    """

    def __init__(self, contract_path: Path):
        self.contract_path = contract_path
        self.data = None
        with self.contract_path.open("r") as file:
            self.data = json.load(file)

    @abstractmethod
    def load_profit_fn(self) -> List[List[ProfitFunction]]:
        """
        Load profit functions from a file.
        """
        NotImplementedError(
            "load_profit_fn() must be implemented in the subclass."
        )

    @abstractmethod
    def load_repetition(self) -> List[int]:
        """
        Load repetition data from a file.
        """
        NotImplementedError(
            "load_repetition() must be implemented in the subclass."
        )
