import json
from pathlib import Path
from typing import List
from scheduler.Profit import ProfitFunction

class ContractGenerator:
    @staticmethod
    def load_profit_fn(file_path: Path) -> List[List[ProfitFunction]]:
        with file_path.open("r") as file:
            data = json.load(file)
        return [[ProfitFunction(**profit_data) for profit_data in job_contracts]
                for job_contracts in data["contracts"].values()]
    
    @staticmethod
    def load_repeatition(file_path: Path) -> List[int]:
        with file_path.open("r") as file:
            data = json.load(file)
        return [len(job_contracts) for job_contracts in data["contracts"].values()]