import pathlib
import tempfile
import traceback
import sys
import streamlit as st

from rl_scheduler.envs.utils import make_env
from rl_scheduler.scheduler import Scheduler


def load_environment(
    machine_file,
    job_file,
    operation_file,
    contract_file,
    seed: int | None = None,
):
    if not all([machine_file, job_file, operation_file, contract_file]):
        return None, "Please upload *all* JSON files before loading the environment."

    try:
        # 1. dump each upload to disk
        m_path = _dump_to_temp(machine_file)
        j_path = _dump_to_temp(job_file)
        o_path = _dump_to_temp(operation_file)
        c_path = _dump_to_temp(contract_file)

        # 2. hand the *paths* to Scheduler
        scheduler = Scheduler(
            machine_config_path=m_path,
            job_config_path=j_path,
            operation_config_path=o_path,
        )

        # 3. create the Gymnasium environment via factory helper
        try:
            env = make_env(scheduler=scheduler)
        except Exception:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            return None, "Error while make_env()\n(see server logs for details)."

        # 4. reset the environment to load the contract
        try:
            env.reset(seed=seed, options={"contract_path": c_path})
        except Exception:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            return None, "Error while env.reset()\n(see server logs for details)."

        return env, None

    except Exception:
        # generic fallback (e.g., dump_to_temp or Scheduler init)
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return None, "Failed to load environment\n(see server logs for details)."

    finally:
        # 3. clean up temporary files (optional: keep them for reuse)
        for p in (m_path, j_path, o_path, c_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def reset_env(env, contract_file):
    """Reset the environment with a new contract."""
    try:
        # 1. dump each upload to disk
        c_path = _dump_to_temp(contract_file)

        # 2. reset the environment with the new contract
        env.reset(options={"contract_path": c_path})

    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        raise RuntimeError(
            "Failed to reset environment\n(see server logs for details)."
        )

    finally:
        # 3. clean up temporary files (optional: keep them for reuse)
        try:
            c_path.unlink(missing_ok=True)
        except Exception:
            pass


def _dump_to_temp(uploaded: st.UploadedFile) -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    tmp.close()
    return pathlib.Path(tmp.name)
