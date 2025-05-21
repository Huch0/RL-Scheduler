# RL-Scheduler: Reinforcement Learning for Repeatable Job Shop Scheduling

## 📌 Introduction
![Scheduling Process Example](images/scheduling_process_12x8.gif)
*An example of the proposed agent's scheduling process.*


**RL-Scheduler** is a research-driven project designed to tackle the **Repeatable Job-Shop Scheduling Problem (RJSP)** using reinforcement learning. In real-world factories, the same job types are repeatedly scheduled under dynamic conditions. To handle this, we integrate a Maskable PPO-based agent with **Estimated Tardiness (ETD)**-driven priority dispatching to optimize job scheduling performance.

A full **Streamlit-based GUI** is included for setting up scheduling environments, configuring agents, training policies, and visualizing results — enabling end-to-end experimentation with no need to modify code manually.

> This project was developed as a 2025 graduation capstone at Pusan National University.

---



## 📂 Project Structure

~~~bash
RL-Scheduler/
├── pyproject.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── rjsp_gui/                   # Streamlit GUI app
│   │   ├── app.py                  # Main GUI entrypoint
│   │   ├── pages/                  # Streamlit page components
│   │   └── services/               # Backend interaction logic
│   ├── rl_scheduler/              # Core RL scheduling logic
│   │   ├── contract_generator/
│   │   ├── envs/                  # RJSP Gym-compatible environments
│   │   ├── gnn/                   # Graph Neural Network (experimental)
│   │   ├── graph/                 # Graph construction tools
│   │   ├── instances/             # Job/Machine JSON config samples
│   │   ├── models/                # Agent save/load utilities
│   │   ├── priority_rule/         # EDD, ETD, EPV-based rules
│   │   ├── renderer/              # Gantt chart rendering (matplotlib/plotly)
│   │   ├── scheduler/             # Core job/machine classes
│   │   ├── tests/                 # Pytest-based validation
│   │   └── trainer/               # Training pipeline via stable-baselines3
│   └── rl_scheduler.egg-info/
└── uv.lock
~~~

## ⚙️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible Python builds.

### Prerequisites

- Python ≥ 3.11
- `uv` installed:

~~~bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~~~

### Setup

Install dependencies:

~~~bash
uv sync
~~~

Launch the GUI:

~~~bash
streamlit run src/rjsp_gui/app.py
~~~

## 🚀 How to Use (via GUI)

The GUI consists of 3 main pages:

| Page             | Tabs (Subsections)                            | Description |
|------------------|-----------------------------------------------|-------------|
| **Scheduler Setup** | `Job`, `Machine`, `Contract`                  | Upload or edit `.json` configuration files for operations, jobs, machines, and contract details. Then export as `scheduler.pkl`. |
| **Model Train**     | `Hyperparameter`, `Handler setup`, `Training`, `Visualization` | Configure PPO parameters, ETD/EDD dispatch rules, reward weighting, and start training. Track live metrics and export models (`env.pkl`, `best_model.zip`). |
| **Playground**      | —                                             | Load a trained agent and scheduler to run simulations interactively or evaluate actions manually/randomly. Export action sequences as `.json`. |

---

## 🧠 Core Techniques

- **ETD (Estimated Tardiness) Priority Rule**  
  A dynamic job urgency metric based on partial progress and estimated delays.

- **Maskable PPO-based agent**  
  Action masking ensures agents never select invalid machine-job pairs, improving stability and convergence.

- **Streamlit GUI Pipeline**  
  End-to-end management from environment building to model export — no code edits required.

- **Multi-scale Training & Visualization**  
  Supports dynamic sampling, Gantt chart rendering, reward tracking, and progress logs.

---

## 🧪 Outputs

After training, each experiment logs the following to `/logs/{AgentName}/{timestamp}/`:

~~~bash
├── best_model.zip
├── env.pkl
├── scheduler.pkl
├── env_config.json
├── train_config.json
├── progress.csv
├── model_checkpoint_*.zip
└── events.out.tfevents.*   # For TensorBoard
~~~

These assets can be reloaded in the GUI's **Playground** tab or used for comparative evaluation.

---

## 📖 Citation

~~~bibtex
@article{heo2024estimated,
  title={Estimated Tardiness-Based Reinforcement Learning Solution to Repeatable Job-Shop Scheduling Problems},
  author={Heo, Chi Yeong and Seo, Jun and Kim, Yonggang and Kim, Yohan and Kim, Taewoon},
  journal={Processes},
  volume={13},
  number={1},
  pages={62},
  year={2024},
  publisher={MDPI}
}
~~~
