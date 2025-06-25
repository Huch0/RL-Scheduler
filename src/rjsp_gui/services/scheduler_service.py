import io
import json
import pickle
import tempfile
from pathlib import Path

import streamlit as st

# ---- Scheduler core imports (project‑specific)
from rl_scheduler.scheduler import Scheduler  # uploaded module
from rl_scheduler.contract_generator.deterministic_generator import DeterministicGenerator  # uploaded module

__all__ = [
    "build_scheduler_pickle",
]

# -------------------------------------------------
# Helper: write session objects → temp JSON files
# -------------------------------------------------

def _dump_json(tmp: Path, name: str, payload: dict | list) -> Path:
    """Dump *payload* as <name>.json inside *tmp* directory and return Path."""
    path = tmp / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    return path


def build_scheduler_pickle() -> io.BytesIO:
    """Create a Scheduler from current Streamlit session & return pickle buffer.

    Raises
    ------
    RuntimeError
        If any of the required session‑state keys are missing.
    """

    required_keys = [
        "machines",
        "operation_templates",
        "job_templates",
        "contracts",
    ]
    if not all(st.session_state.get(k) for k in required_keys):
        raise RuntimeError(
            "🛠️  Machine, Job, Operation, Contract 설정이 모두 필요합니다."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # 1. dump configs
        mach_p = _dump_json(tmp, "machines", {"machines": st.session_state.machines})
        job_p = _dump_json(tmp, "jobs", {"jobs": st.session_state.job_templates})
        op_p = _dump_json(
            tmp, "operations", {"operations": st.session_state.operation_templates}
        )
        con_p = _dump_json(tmp, "contracts", {"contracts": st.session_state.contracts})

        # 2. create Scheduler
        sched = Scheduler(
            machine_config_path=mach_p,
            job_config_path=job_p,
            operation_config_path=op_p,
        )

        # 3. apply deterministic generator for repetition & profit fn
        deterministicGenerator = DeterministicGenerator(contract_path=con_p)
        repetitions = deterministicGenerator.load_repetition()
        profit_fn = deterministicGenerator.load_profit_fn()
        sched.reset(repetitions=repetitions, profit_functions=profit_fn)

        # 4. pickle to BytesIO
        buf = io.BytesIO()
        pickle.dump(sched, buf)
        buf.seek(0)
        return buf
