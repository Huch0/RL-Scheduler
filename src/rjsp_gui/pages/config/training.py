import json, yaml
import streamlit as st
from .utils import ensure_state

# ────────────────────────────────────────────────
#  Training Hyperparameter Configuration Page
# ────────────────────────────────────────────────

def render_training_config() -> None:
    """UI for collecting SB3‑style hyperparameters and exporting YAML."""
    ensure_state()
    st.header("Training Hyperparameters")

    # ───────────── Model + Net architecture ─────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Architecture")
        model_type = st.selectbox("Model Type", ["DQN", "PPO", "A2C", "DDPG"], key="model_type")
        hidden_layers_str = st.text_input("Hidden Layers (comma)", "256,128,64", key="hidden_layers")
        activation = st.selectbox("Activation", ["relu", "tanh", "sigmoid", "leaky_relu"], key="act")

    with col2:
        st.subheader("Training Parameters")
        lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%f", key="lr")
        batch_size = st.number_input("Batch Size", 16, 1024, 64, 16, key="batch")
        n_episodes = st.number_input("Number of Episodes", 100, 100000, 10000, 1000, key="epi")
        gamma = st.slider("Gamma", 0.8, 0.999, 0.99, 0.01, key="gamma")

    # ───────────── Advanced params ─────────────
    st.subheader("Advanced Parameters")
    col3, col4 = st.columns(2)

    with col3:
        exploration = st.selectbox("Exploration", ["epsilon_greedy", "boltzmann", "ucb"], key="explore")
        eps_cfg = {}
        if exploration == "epsilon_greedy":
            eps_cfg["epsilon_start"] = st.number_input("Epsilon Start", 0.1, 1.0, 1.0, 0.1, key="eps_s")
            eps_cfg["epsilon_end"]   = st.number_input("Epsilon End", 0.01, 0.5, 0.05, 0.01, key="eps_e")
            eps_cfg["epsilon_decay"] = st.number_input("Epsilon Decay", 500, 20000, 10000, 500, key="eps_d")

    with col4:
        other_cfg = {}
        if model_type in ["DQN", "DDPG"]:
            other_cfg["target_update_interval"] = st.number_input("Target Net Update", 100, 10000, 1000, 100, key="target")
        if model_type == "PPO":
            other_cfg["clip_ratio"] = st.number_input("Clip Ratio", 0.1, 0.5, 0.2, 0.05, key="clip")
            other_cfg["gae_lambda"] = st.number_input("GAE Lambda", 0.8, 1.0, 0.95, 0.01, key="gae")

    # ───────────── YAML export ─────────────
    st.divider()
    if st.button("Generate YAML", key="gen_yaml"):
        # parse layers
        try:
            layers = [int(x.strip()) for x in hidden_layers_str.split(",") if x.strip()]
        except ValueError:
            st.error("Hidden layers 입력이 잘못되었습니다.")
            return

        cfg: dict = {
            "algorithm": model_type,
            "policy": "MlpPolicy",
            "learning_rate": float(lr),
            "batch_size": int(batch_size),
            "gamma": float(gamma),
            "n_episodes": int(n_episodes),
            "net_arch": layers,
            "activation_fn": activation,
            "exploration": exploration,
        }
        cfg.update(eps_cfg)
        cfg.update(other_cfg)

        yaml_text = yaml.safe_dump(cfg, sort_keys=False)
        st.code(yaml_text, language="yaml")
        st.download_button("Download YAML", data=yaml_text, file_name="sb3_config.yaml", mime="text/yaml")

    # ℹ️  SB3 note
    st.info("Stable‑Baselines3는 기본적으로 YAML 로딩을 지원하지 않지만, 아래와 같이 Python 코드에서 쉽게 읽어 사용할 수 있습니다:\n\n"  \
            "```python\nimport yaml, stable_baselines3 as sb3\nwith open('sb3_config.yaml') as f:\n    cfg = yaml.safe_load(f)\nmodel_cls = getattr(sb3, cfg['algorithm'])\nmodel = model_cls(cfg['policy'], env, **{k:v for k,v in cfg.items() if k not in ['algorithm','policy']})\n```")
