import pytest
from scheduler.Factory.TemplateLoader import TemplateLoader
from config_path import INSTANCES_DIR

machine_example_path = INSTANCES_DIR / "Machines/M-example0-3.json"
job_example_path = INSTANCES_DIR / "Jobs/J-example0-5.json"
operation_example_path = INSTANCES_DIR / "Operations/O-example0.json"


@pytest.fixture
def machine_templates():
    return TemplateLoader.load_machine_templates(machine_example_path)


@pytest.fixture
def job_templates():
    return TemplateLoader.load_job_templates(job_example_path)


@pytest.fixture
def operation_templates():
    return TemplateLoader.load_operation_templates(operation_example_path)


def test_load_machine_templates(machine_templates):
    print()
    for mt in machine_templates:
        print(mt)


def test_load_job_templates(job_templates):
    print()
    for jt in job_templates:
        print(jt)


def test_load_operation_templates(operation_templates):
    print()
    for ot in operation_templates:
        print(ot)
