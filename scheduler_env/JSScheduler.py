import random
import numpy as np
import copy


class Job():
    max_n_operations = 3

    def __init__(self, job_info):
        self.id = job_info['id']
        self.color = '#' + ''.join(random.choices('0123456789ABCDEF', k=6))

        self.operation_queue = [Operation(op_info) for op_info in job_info['operation_queue']]
        # Fill the operation queue with dummy operations
        if len(self.operation_queue) < self.max_n_operations:
            self.operation_queue += [Operation({'id': -1, 'type': -1, 'processing_time': -1, 'predecessor': -1})] * (
                self.max_n_operations - len(self.operation_queue))

        # self.deadline = job_info['deadline']

    def __str__(self):
        return f"Job {self.id}"

    def encode(self):
        return [op.encode() for op in self.operation_queue]


class Operation():
    def __init__(self, op_info):
        self.id = op_info['id']
        self.type = op_info['type']
        self.processing_time = op_info['processing_time']
        self.predecessor = op_info['predecessor']

        # Informations for runtime
        self.start_time = None
        self.end_time = None
        self.machine = None

    def __str__(self):
        return f"Operation {self.id}, type: {self.type}, processing_time: {self.processing_time}"

    def encode(self):
        return [self.id, self.type, self.processing_time, self.predecessor]


class Machine():
    def __init__(self, machine_info):
        self.id = machine_info['id']
        self.ability = machine_info['ability']

        # Informations for runtime
        self.utilization = 0

    def __str__(self):
        return f"Machine {self.id}, ability: {self.ability}"

    def can_process_op(self, op_type):
        return op_type in self.ability


class JSScheduler():
    def __init__(self, job_instance, machine_instance, max_job_repetition=3, seed=0):
        self.jobs = [Job(job_info) for job_info in job_instance]
        self.machines = [Machine(machine_info) for machine_info in machine_instance]

        self.n_jobs = len(self.jobs)
        self.n_machines = len(self.machines)
        self.max_job_repetition = max_job_repetition

        # (# of jobs) x (max # of job repetition) x (max # of operations per job)
        self.job_buffer = self._fill_job_buffer()

    def _fill_job_buffer(self):
        # Repeat each jobs randomly
        np.random.seed(0)
        job_repetition = np.random.randint(
            1, self.max_job_repetition + 1, self.n_jobs)

        job_buffer = [[] for _ in range(self.n_jobs)]
        for i in range(self.n_jobs):
            for _ in range(job_repetition[i]):
                job_buffer[i].append(copy.deepcopy(self.jobs[i]))

            # Fill the job buffer with dummy jobs
            if len(job_buffer[i]) < self.max_job_repetition:
                job_buffer[i] += [Job({'id': -1, 'operation_queue': []})] * \
                    (self.max_job_repetition - len(job_buffer[i]))

        return job_buffer
