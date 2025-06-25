from unittest.mock import MagicMock
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for some backends
import matplotlib.figure as mpl_fig
from rl_scheduler.renderer.matplotlib_renderer import MatplotRenderer


def test_gantt_renderer():
    scheduler = MagicMock()
    machine1 = MagicMock()
    machine2 = MagicMock()

    fake_op1 = MagicMock()
    fake_op1.start_time = 0
    fake_op1.end_time = 3
    fake_op1.type_code = "OpTypeA"
    fake_op1.job_instance.job_template.job_template_id = 0
    fake_op1.job_instance.job_instance_id = 0
    fake_op1.job_instance.color = (1.0, 0.6, 0.6, 0.8)

    fake_op2 = MagicMock()
    fake_op2.start_time = 3
    fake_op2.end_time = 7
    fake_op2.type_code = "OpTypeB"
    fake_op2.job_instance.job_template.job_template_id = 0
    fake_op2.job_instance.job_instance_id = 1
    fake_op2.job_instance.color = (1.0, 0.6, 0.6, 1.0)

    fake_op3 = MagicMock()
    fake_op3.start_time = 1
    fake_op3.end_time = 4
    fake_op3.type_code = "OpTypeC"
    fake_op3.job_instance.job_template.job_template_id = 1
    fake_op3.job_instance.job_instance_id = 0
    fake_op3.job_instance.color = (0.6, 1.0, 0.6, 0.8)

    fake_op4 = MagicMock()
    fake_op4.start_time = 5
    fake_op4.end_time = 10
    fake_op4.type_code = "OpTypeD"
    fake_op4.job_instance.job_template.job_template_id = 2
    fake_op4.job_instance.job_instance_id = 0
    fake_op4.job_instance.color = (0.6, 0.6, 1.0, 0.8)

    machine1.assigned_operations = [fake_op1, fake_op2]
    machine2.assigned_operations = [fake_op3, fake_op4]

    scheduler.machine_instances = [machine1, machine2]

    fig = MatplotRenderer.render(scheduler, title="Matplotlib Gantt Chart")
    assert isinstance(fig, mpl_fig.Figure)
