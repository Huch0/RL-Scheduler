# filepath: c:\Users\USER\Desktop\4-1\RL-Scheduler\rl-scheduler\tests\renderer\test_renderer.py
import pytest
from pathlib import Path
from renderer.Renderer import Renderer
from unittest.mock import MagicMock
from config_path import INSTANCES_DIR


def test_render_gantt():
    machine1 = MagicMock()
    machine2 = MagicMock()

    fake_op1 = MagicMock()
    fake_op1.start_time = 0
    fake_op1.end_time = 3
    fake_op1.type_code = "OpTypeA"
    fake_op1.job_instance.job_template.job_template_id = 0
    fake_op1.job_instance.job_instance_id = 0

    fake_op2 = MagicMock()
    fake_op2.start_time = 3
    fake_op2.end_time = 7
    fake_op2.type_code = "OpTypeB"
    fake_op2.job_instance.job_template.job_template_id = 0
    fake_op2.job_instance.job_instance_id = 1

    fake_op3 = MagicMock()
    fake_op3.start_time = 1
    fake_op3.end_time = 4
    fake_op3.type_code = "OpTypeC"
    fake_op3.job_instance.job_template.job_template_id = 1
    fake_op3.job_instance.job_instance_id = 0

    fake_op4 = MagicMock()
    fake_op4.start_time = 5
    fake_op4.end_time = 10
    fake_op4.type_code = "OpTypeD"
    fake_op4.job_instance.job_template.job_template_id = 2
    fake_op4.job_instance.job_instance_id = 0

    machine1.assigned_operations = [fake_op1, fake_op2]
    machine2.assigned_operations = [fake_op3, fake_op4]

    render_info_path = INSTANCES_DIR / "RenderInfos" / "R-example0.json"

    # 1) Matplotlib Gantt
    Renderer.render_gantt(
        machine_instances=[machine1, machine2],
        render_info_path=render_info_path,
        title="Matplotlib Gantt Chart"
    )

    # 2) Plotly Interactive Gantt
    Renderer.render_gantt_interactive(
        machine_instances=[machine1, machine2],
        render_info_path=render_info_path,
        title="Plotly Interactive Gantt"
    )

    assert True