# src/rjsp_gui/services/train_service.py
from __future__ import annotations
import io, json, pickle, tempfile, zipfile
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from rl_scheduler.envs.utils import make_env
from rl_scheduler.trainer.trainer import load_sb3_algo, train_agent
from rl_scheduler.envs.rjsp_env import RJSPEnv

__all__ = ["build_env", "train_model", "make_run_dir", "zip_artifacts"]

# ───────────────────────── helpers ──────────────────────────
def make_run_dir(root: Path, agent: str) -> Path:
    run_dir = root / agent / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def zip_artifacts(run_dir: Path) -> io.BytesIO:
    """run_dir 내 결과물을 하나의 zip 버퍼로 반환"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in run_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".zip", ".csv", ".npz",
                                                    ".json", ".pkl"}:
                zf.write(p, p.relative_to(run_dir))
    buf.seek(0)
    return buf

# ────────────────── environment builder ─────────────────────
def build_env(
    scheduler_buf: io.BytesIO,
    contract_file: Any | None,          # dict | file-like | Path | None
    *,
    action_handler: str | None,
    observation_handler: str | None,
    reward_handler: str | None,
    info_handler: str | None,
) -> tuple[RJSPEnv, Path]:
    """
    Returns (env, contract_path)
    """
    scheduler = pickle.loads(scheduler_buf.getvalue())

    # contract JSON → tmp 폴더 영구화
    tmp_dir        = Path(tempfile.mkdtemp(prefix="rjsp_"))
    contract_path  = tmp_dir / "contract.json"
    if isinstance(contract_file, dict):
        contract_data = contract_file
    elif hasattr(contract_file, "read"):
        contract_data = json.load(contract_file)
    elif contract_file:
        contract_data = json.loads(Path(contract_file).read_text())
    else:
        contract_data = {"contracts": {}}
    contract_path.write_text(json.dumps(contract_data, indent=4))

    cg_name = "stochastic" if "sampling" in contract_data else "deterministic"

    env = make_env(
        scheduler              = scheduler,
        contract_generator     = cg_name,
        contract_path          = contract_path,
        action_handler         = action_handler,
        action_handler_kwargs  = {"priority_rule_id": "etd"},
        observation_handler    = observation_handler,
        observation_handler_kwargs = {"time_horizon": 1000},
        reward_handler         = reward_handler,
        reward_handler_kwargs  = {"weights": {"C_max": 10., "C_mup": 2., "C_mid": 1.}},
        info_handler           = info_handler,
    )
    env.reset()
    return env, contract_path

# ───────────────────── 학습 실행기 ───────────────────────────
def train_model(
    *,
    scheduler_buf: io.BytesIO,
    contract_file: Any | None,
    hp_cfg: Dict[str, Any],          # {"algorithm": str, "hyperparameters": {...}}
    agent_name: str,
    log_root: Path,
    total_steps: int,
    eval_freq: int,
    ckpt_freq: int,
    action_handler: str | None = None,
    observation_handler: str | None = None,
    reward_handler: str | None = None,
    info_handler: str | None = None,
    num_envs: int = 1,
    max_epi_steps: int = 1_000,
) -> Path:
    """
    End-to-end trainer (blocking). 반환값: run_dir Path
    """
    # 1) run 디렉터리
    run_dir = make_run_dir(log_root, agent_name)

    # 2) 환경 생성
    env, contract_path = build_env(
        scheduler_buf        = scheduler_buf,
        contract_file        = contract_file,
        action_handler       = action_handler,
        observation_handler  = observation_handler,
        reward_handler       = reward_handler,
        info_handler         = info_handler,
    )

    # 3) env.pkl 저장
    env.save(run_dir)                         # env.pkl
    #    scheduler.pkl (재현성을 위해)
    with (run_dir / "scheduler.pkl").open("wb") as f:
        pickle.dump(env.scheduler, f)

    # Extract scheduler template data for embedding as serializable dictionaries
    scheduler_templates = env.scheduler.get_templates_as_dicts()
    
    # contract.json 파일의 내용을 직접 로드
    contract_content = json.loads(contract_path.read_text())
    
    env_config: Dict[str, Any] = {
        "scheduler": scheduler_templates,
        "contract_generator": "stochastic" if isinstance(contract_file, dict) else "deterministic",
        "contracts": contract_content,
        "action_handler": action_handler,
        "action_handler_kwargs": {"priority_rule_id": "etd"},
        "observation_handler": observation_handler,
        "observation_handler_kwargs": {"time_horizon": 1000},
        "reward_handler": reward_handler,
        "reward_handler_kwargs": {"weights": {"C_max": 10.0, "C_mup": 2.0, "C_mid": 1.0}},
        "info_handler": info_handler,
    }
    (run_dir / "env_config.json").write_text(json.dumps(env_config, indent=4))

    # 5) train_config.json 저장
    train_cfg = {
        "agent_name": agent_name,
        "ALGO": hp_cfg["algorithm"],
        "total_timesteps": total_steps,
        "checkpoint_freq": ckpt_freq,
        "eval_freq": eval_freq,
        "num_envs": num_envs,
        "max_episode_steps": max_epi_steps,
        "hyperparameters": hp_cfg["hyperparameters"],
    }
    (run_dir / "train_config.json").write_text(json.dumps(train_cfg, indent=4))

    # 6) 학습 실행

    from stable_baselines3 import PPO, A2C, DQN, DDPG
    from sb3_contrib import MaskablePPO
    ALGO_TABLE = {
        "PPO": PPO,
        "MaskablePPO": MaskablePPO,
        "DQN": DQN,
        "A2C": A2C,
        "DDPG": DDPG,
    }
    ALGO = ALGO_TABLE.get(hp_cfg["algorithm"])

    train_agent(
        env                 = env,
        save_dir            = str(run_dir),
        ALGO                = ALGO,
        total_timesteps     = total_steps,
        checkpoint_freq     = ckpt_freq,
        num_eval_episodes   = 5,
        eval_freq           = eval_freq,
        hyperparameters     = hp_cfg["hyperparameters"],
        num_envs            = num_envs,
        max_episode_steps   = max_epi_steps,
    )
        
    return run_dir
