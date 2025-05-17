import json
from pathlib import Path
from typing import List
from rl_scheduler.scheduler.profit import ProfitFunction
from .contract_generator import ContractGenerator


class DeterministicGenerator(ContractGenerator):
    def __init__(self, contract_path: Path):
        super().__init__(contract_path)

    def load_profit_fn(self) -> List[List[ProfitFunction]]:
        with self.contract_path.open("r") as file:
            data = json.load(file)
        return [
            [ProfitFunction(**profit_data) for profit_data in job_contracts]
            for job_contracts in data["contracts"].values()
        ]

    def load_repetition(self) -> List[int]:
        with self.contract_path.open("r") as file:
            data = json.load(file)
        return [len(job_contracts) for job_contracts in data["contracts"].values()]
