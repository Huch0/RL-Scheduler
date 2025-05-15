from typing import Dict, List, Any
from pathlib import Path
import json, random

from rl_scheduler.scheduler.profit import ProfitFunction
from rl_scheduler.contract_generator.contract_generator import ContractGenerator

__all__ = ["StochasticGenerator"]

class StochasticGenerator(ContractGenerator):
    """
    Generate contracts by sampling repetition and deadline per job template.
    sampling_config format:
    {
      "<job_id>": {"repetition": {"mean": m, "std": s},
                     "deadline":   {"mean": m2, "std": s2}}
    }
    base_contracts should be dict from deterministic JSON: {"job_<id>": [{...}, ...]}
    """
    # cache to store last samples per sampling file
    _cache: Dict[str, Dict[str, Any]] = {}

    def load_repetition(self) -> List[int]:
        """
        Sample repetition counts per job from sampling JSON.
        JSON format: { "sampling": { "<job_id>": {"repetition": {"mean": m, "std": s}, ...} } }
        Returns list of repetition counts in order of sorted job IDs.
        """
        # Sample repetition, deadlines, price, and penalties; store in cache
        data = json.load(self.contract_path.open('r'))
        sampling = data.get('sampling', {})
        reps: List[int] = []
        deadlines: List[List[int]] = []
        prices: List[List[float]] = []
        penalties: List[List[float]] = []
        # sample for each job in sorted order
        for job_key in sorted(sampling, key=lambda k: int(k)):
            params = sampling[job_key]
            # repetition sampling
            rep_params = params.get('repetition', {})
            mean_r, std_r = rep_params.get('mean', 1), rep_params.get('std', 0)
            count = max(1, int(random.gauss(mean_r, std_r)))
            reps.append(count)
            # deadline sampling
            dl_params = params.get('deadline', {})
            mean_d, std_d = dl_params.get('mean', 1), dl_params.get('std', 0)
            dlist = [max(1, int(random.gauss(mean_d, std_d))) for _ in range(count)]
            deadlines.append(dlist)
            # price sampling
            price_params = params.get('price', {})
            mean_p, std_p = price_params.get('mean', 0.0), price_params.get('std', 0.0)
            plist = [max(0.0, random.gauss(mean_p, std_p)) for _ in range(count)]
            prices.append(plist)
            # late_penalty sampling
            pen_params = params.get('late_penalty', {})
            mean_pen, std_pen = pen_params.get('mean', 0.0), pen_params.get('std', 0.0)
            penlist = [max(0.0, random.gauss(mean_pen, std_pen)) for _ in range(count)]
            penalties.append(penlist)
        # cache samples
        key = str(self.contract_path.resolve())
        StochasticGenerator._cache[key] = {
            'repetition': reps,
            'deadlines': deadlines,
            'prices': prices,
            'penalties': penalties,
        }
        return reps

    def load_profit_fn(self) -> List[List[ProfitFunction]]:
        # Construct profit functions using cached samples
        key = str(self.contract_path.resolve())
        state = StochasticGenerator._cache.get(key, {})
        reps = state.get('repetition', [])
        deadlines = state.get('deadlines', [])
        prices = state.get('prices', [])
        penalties = state.get('penalties', [])
        profs: List[List[ProfitFunction]] = []
        for idx, count in enumerate(reps):
            dlist = deadlines[idx] if idx < len(deadlines) else []
            plist = prices[idx] if idx < len(prices) else []
            penlist = penalties[idx] if idx < len(penalties) else []
            job_funcs: List[ProfitFunction] = []
            for i in range(count):
                d = dlist[i] if i < len(dlist) else 0
                p = plist[i] if i < len(plist) else 0.0
                pen = penlist[i] if i < len(penlist) else 0.0
                job_funcs.append(ProfitFunction(job_instance_id=i, price=p, deadline=d, late_penalty=pen))
            profs.append(job_funcs)
        return profs
