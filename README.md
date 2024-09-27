# RL-Scheduler: Reinforcement Learning for Repeatable Job Shop Scheduling

## Introduction
![스케줄링 프로세스](images/scheduling_process_12x8.gif)
*An example of the proposed agent's scheduling process.*

RL-Scheduler is a project focused on applying Reinforcement Learning (RL) techniques to the complex problem of Repeatable Job Shop Scheduling. By modeling the scheduling environment as a custom Gym environment (RJSPEnv), we leverage state-of-the-art RL algorithms, specifically **MaskablePPO**, to learn efficient scheduling policies that can handle dynamic and complex job shop scenarios with invalid actions.

## Project Structure
~~~ bash
RL-Scheduler/
├── README.md
├── RJSPEnv/
│   ├── Env.py
│   └── Scheduler.py
├── instances/
│   ├── Jobs/
│   │   ├── v0-12x8-12.json
│   │   └── ...
│   └── Machines/
│       ├── v0-12x8.json
│       └── ...
├── models/
│   └── paper/
│       ├── 0-paper-8x12-18m/
│           ├── MP_Single_Env4_gamma_1_obs_v4_clip_1_lr_custom_expv1_18000000.zip
│           └── ...
├── tutorial.ipynb
└── requirements.txt
~~~
- RJSPEnv/: Contains the environment (Env.py) and scheduler (Scheduler.py) code.
- instances/: Contains job and machine configuration files.
- models/: Pre-trained models and training logs.
- tutorial.ipynb: Notebook demonstrating how to use the pre-trained model.
- README.md: Project documentation.
- requirements.txt: Required Python packages.


## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Install Required Packages

1.	Create a virtual environment:
~~~bash
python -m venv venv
~~~

2.	Activate the virtual environment:
-   On Windows:
~~~bash
venv\Scripts\activate
~~~
-   On MacOS/Linux:
~~~bash
source venv/bin/activate
~~~

3.	Upgrade pip and install required packages:
~~~bash
pip install --upgrade pip
pip install -r requirements.txt
~~~

## How To Use

### Running the Tutorial

Follow the tutorial.ipynb notebook to learn how to load and test the pre-trained MaskablePPO model.

~~~bash
jupyter notebook tutorial.ipynb
~~~

## Target Audience

This project is intended for researchers, students, and practitioners interested in applying reinforcement learning to scheduling problems, especially where invalid actions need to be handled effectively. A background in machine learning and familiarity with RL concepts is recommended.
