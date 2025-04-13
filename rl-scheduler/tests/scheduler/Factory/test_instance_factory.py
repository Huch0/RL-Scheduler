from scheduler.Factory.TemplateLoader import TemplateLoader
from contract_generator import DeterministicGenerator
from scheduler.Factory.InstanceFactory import InstanceFactory
from config_path import INSTANCES_DIR


def test_machine_instances():
    # Test that machine instances are correctly created.
    machine_example_path = INSTANCES_DIR / "Machines/M-example0-3.json"
    machine_templates = TemplateLoader.load_machine_templates(machine_example_path)
    factory = InstanceFactory(machine_templates, [], [])
    machine_instances = factory.get_new_machine_instances()
    assert len(machine_instances) == len(machine_templates)
    print("Machine Instances:")
    for mi in machine_instances:
        print(mi)


def test_job_instances():
    # Test that job instances are correctly created.
    job_example_path = INSTANCES_DIR / "Jobs/J-example0-5.json"
    operation_example_path = INSTANCES_DIR / "Operations/O-example0.json"
    contract_path = INSTANCES_DIR / "Contracts/C-example0-5.json"
    job_templates = TemplateLoader.load_job_templates(job_example_path)
    operation_templates = TemplateLoader.load_operation_templates(
        operation_example_path
    )
    profit_functions = DeterministicGenerator.load_profit_fn(contract_path)
    repetitions = DeterministicGenerator.load_repetition(contract_path)

    factory = InstanceFactory([], operation_templates, job_templates)
    job_instances = factory.get_new_job_instances(repetitions, profit_functions)

    assert sum([len(jobs) for jobs in job_instances]) == sum(repetitions)
    print("Job Instances:")
    i = 0
    for group in job_instances:
        print("Job {}:".format(i))
        i += 1
        for ji in group:
            print(ji)


def test_operation_instances():
    # Test that each job instance has a properly chained sequence of operation instances.
    job_example_path = INSTANCES_DIR / "Jobs/J-example0-5.json"
    operation_example_path = INSTANCES_DIR / "Operations/O-example0.json"
    contract_path = INSTANCES_DIR / "Contracts/C-example0-5.json"
    job_templates = TemplateLoader.load_job_templates(job_example_path)
    operation_templates = TemplateLoader.load_operation_templates(
        operation_example_path
    )
    profit_functions = DeterministicGenerator.load_profit_fn(contract_path)
    repetitions = DeterministicGenerator.load_repetition(contract_path)

    factory = InstanceFactory([], operation_templates, job_templates)
    job_instances = factory.get_new_job_instances(repetitions, profit_functions)

    print("Operation chain for Job Instances:")
    for group in job_instances:
        for job in group:
            ops = job.operation_instance_sequence
            # Check the first operation has no predecessor.
            assert ops[0].predecessor is None
            # Verify each subsequent operation is chained.
            for i in range(1, len(ops)):
                assert ops[i].predecessor == ops[i - 1]
            for i in range(len(ops) - 1):
                assert ops[i].successor == ops[i + 1]
            # Print the operation chain clearly for debugging.
            # chain_str = " -> ".join([f"[id: {op.operation_template.operation_template_id} | pred: {'None' if op.predecessor is None else id(op.predecessor)} | succ: {'None' if op.successor is None else id(op.successor)}]" for op in ops])
            # print(f"Job instance {job.job_instance_id} operation chain:")
            # print(chain_str)
# pytest에서 print 내용을 보려면 터미널에서 "pytest -s" 옵션으로 실행하세요.
