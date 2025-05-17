import sys
import traceback
from rl_scheduler.envs.registry import get_action_handler
from rl_scheduler.envs.rjsp_env import RJSPEnv
from rl_scheduler.envs.utils import make_env
from rl_scheduler.scheduler import Scheduler
from .utils import dump_to_temp


def load_environment(
    machine_file,
    job_file,
    operation_file,
    contract_file,
    seed: int | None = None,
):
    if not all([machine_file, job_file, operation_file, contract_file]):
        return None, (
            "Please upload *all* JSON files before loading the " "environment."
        )

    try:
        # 1. dump each upload to disk
        m_path = dump_to_temp(machine_file)
        j_path = dump_to_temp(job_file)
        o_path = dump_to_temp(operation_file)
        c_path = dump_to_temp(contract_file)

        # 2. hand the *paths* to Scheduler
        scheduler = Scheduler(
            machine_config_path=m_path,
            job_config_path=j_path,
            operation_config_path=o_path,
        )

        # 3. create the Gymnasium environment via factory helper
        try:
            env = make_env(scheduler=scheduler, contract_path=c_path)
        except Exception:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            return None, ("Error while make_env()\n(see server logs for " "details).")

        # 4. reset the environment to load the contract
        try:
            env.contract_generator.contract_path = c_path
            env.reset(seed=seed)
        except Exception:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)
            return None, ("Error while env.reset()\n(see server logs for " "details).")

        return env, None

    except Exception:
        # generic fallback (e.g., dump_to_temp or Scheduler init)
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return None, ("Failed to load environment\n(see server logs for " "details).")

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
        if contract_file:
            # 1. dump each upload to disk
            c_path = dump_to_temp(contract_file)

            # 2. reset the environment with the new contract
            env.contract_generator.contract_path = c_path
        env.reset()

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


def step_env(
    env: RJSPEnv,
    action_handler_id: str,
    action: tuple[int, ...],
):
    """
    Execute one environment step using a dynamically‑selected action handler.

    The caller supplies **`action_handler_id`**, a string key understood by
    `rl_scheduler.envs.registry.get_action_handler()`.  This helper
    instantiates
    the correct `ActionHandler` subclass, assigns it to `env.action_handler`,
    updates `env.action_space`, and finally forwards *action* to
    `env.step(action)`.

    Parameters
    ----------
    env : RJSPEnv
        A running RJSP environment whose scheduler state will be advanced.
    action_handler_id : str
        Lookup key passed to `get_action_handler()` (e.g. ``"mjr"`` for the
        default MJR handler).  The resulting handler must accept the provided
        *action*.
    action : tuple[int, ...]
        A tuple of integers drawn from the new handler’s ``action_space``.
        Length is handler‑dependent; the function raises ``ValueError`` if the
        type does not match.

    Returns
    -------
    tuple
        Whatever tuple your Gymnasium version returns from ``env.step``—
        typically ``(obs, reward, terminated, truncated, info)``.

    Raises
    ------
    RuntimeError
        Propagated when handler instantiation or ``env.step`` fails.
        The full traceback is printed to *stderr* for developer diagnosis,
        while a concise message surfaces to the Streamlit UI.
    """
    # Ensure action is a tuple of ints (any length)
    if not isinstance(action, tuple) or not all(isinstance(x, int) for x in action):
        raise ValueError(
            "Parameter 'action' must be a tuple of integers, e.g. (machine_id, job_id, repetition)"
        )

    try:
        action_handler = get_action_handler(action_handler_id, env.scheduler)
        env.action_handler = action_handler
        env.action_space = action_handler.action_space

        return env.step(action)

    except Exception:

        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        raise RuntimeError(
            "Failed to perform step() with the provided action tuple "
            "(see server logs for details)."
        )
