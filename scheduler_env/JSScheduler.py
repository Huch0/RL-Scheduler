import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import cycle


class Job():
    max_n_operations = 3
    color_generator = None

    def _color_generator(self):
        # Define a list of colors that are distinct and visually appealing on a Gantt chart
        colors = [
            '#1f77b4',  # Muted blue
            '#ff7f0e',  # Safety orange
            '#2ca02c',  # Cooked asparagus green
            '#d62728',  # Brick red
            '#9467bd',  # Muted purple
            '#8c564b',  # Chestnut brown
            '#e377c2',  # Raspberry yogurt pink
            '#7f7f7f',  # Middle gray
            '#bcbd22',  # Curry yellow-green
            '#17becf'   # Blue-teal
        ]
        for color in cycle(colors):
            yield color

    def __init__(self, job_info):
        if Job.color_generator is None:
            Job.color_generator = self._color_generator()

        self.id = job_info['id']
        self.color = next(Job.color_generator)

        self.operation_queue = [Operation(op_info, self.id) for op_info in job_info['operation_queue']]
        # Fill the operation queue with dummy operations
        if len(self.operation_queue) < self.max_n_operations:
            self.operation_queue += [Operation({'id': -1, 'type': -1, 'processing_time': -1, 'predecessor': -1}, self.id)] * (
                self.max_n_operations - len(self.operation_queue))

        # self.deadline = job_info['deadline']

    def __repr__(self):
        return f"Job {self.id}"

    def encode(self):
        encoded_op_queue = np.array([op.encode() for op in self.operation_queue])
        if len(self.operation_queue) < self.max_n_operations:
            encoded_op_queue = np.concatenate((encoded_op_queue, np.full(
                (self.max_n_operations - len(self.operation_queue), 6), -1)))

        return encoded_op_queue


class Operation():
    def __init__(self, op_info, job_id):
        self.id = op_info['id']
        self.type = op_info['type']
        self.processing_time = op_info['processing_time']
        self.predecessor = op_info['predecessor']
        if self.predecessor == None:
            self.predecessor = -1
        self.job_id = job_id

        # Informations for runtime
        self.start_time = -1
        self.end_time = -1
        self.machine = -1

    def __repr__(self):
        return f"OP{self.id} - Job{self.job_id}, {self.type}, {self.processing_time}, {self.start_time}, {self.end_time}"

    def encode(self):
        return np.array([self.id, self.type, self.processing_time, self.predecessor, self.start_time, self.end_time])


class Machine():
    max_n_ability = 3

    def __init__(self, machine_info):
        self.id = machine_info['id']
        self.ability = machine_info['ability']

        # Informations for runtime
        self.utilization = float(0)
        self.last_time = 0
        self.up_time = 0

        # for human visualization
        self.scheduled_ops = []

    def __repr__(self):
        return f"Machine {self.id}, ability: {self.ability}"

    def encode(self):
        ability = []
        if len(self.ability) < self.max_n_ability:
            ability = self.ability + [-1] * (self.max_n_ability - len(self.ability))

        return np.concatenate(([self.id], ability, [self.utilization, self.last_time]))

    def can_process_op(self, op_type):
        return op_type in self.ability


class JSScheduler():
    def __init__(self, job_config: dict, machine_config: dict, max_job_repetition: int = 3, seed=0):
        self.original_machine_list = [Machine(machine_info) for machine_info in machine_config['machines']]
        self.machines = copy.deepcopy(self.original_machine_list)

        self.jobs = [Job(job_info) for job_info in job_config['jobs']]
        self.operations = [op for job in self.jobs for op in job.operation_queue]

        self.n_jobs = job_config['n_jobs']
        self.n_machines = machine_config['n_machines']
        self.total_ops = len(self.operations)

        self.max_job_repetition = max_job_repetition

        self.seed = seed
        # Job buffer is a 4D array that stores remaining operations of each job
        self.job_buffer = np.full((self.n_jobs, self.max_job_repetition, Job.max_n_operations, 6), -1, dtype=np.int16)
        self.job_cursors = [0] * self.n_jobs
        self.op_cursors = [0] * self.n_jobs
        self.job_infos = [] * self.n_jobs  # for human readability
        self.job_repetition = []
        self._fill_job_buffer_job_infos()

        # Schedule table is a 3D array that stores scheduled operations
        self.schedule_table = np.full((self.n_machines, self.total_ops *
                                      self.max_job_repetition, 6), -1, dtype=np.int16)
        self.schedule_table_indices = np.zeros((self.n_machines), dtype=int)
        self.global_last_time = 0

        self.machine_info = np.array([machine.encode() for machine in self.machines], dtype=np.float32)

        self.valid_actions = self._get_valid_actions()

    def reset(self):
        # Reset PRNG
        np.random.seed(self.seed)

        self.machines = copy.deepcopy(self.original_machine_list)
        self._fill_job_buffer_job_infos()
        self.job_cursors = [0] * self.n_jobs
        self.op_cursors = [0] * self.n_jobs
        self.schedule_table = np.full((self.n_machines, self.total_ops *
                                      self.max_job_repetition, 6), -1, dtype=np.int16)
        self.schedule_table_indices = np.zeros((self.n_machines), dtype=int)
        self.machine_info = np.array([machine.encode() for machine in self.machines], dtype=np.float32)
        self.valid_actions = self._get_valid_actions()

        # reset machine

    def _fill_job_buffer_job_infos(self):
        # (# of jobs) x (max # of job repetition) x (max # of operations per job) x (operation info)
        # Repeat each jobs randomly
        np.random.seed(self.seed)
        self.job_repetition = np.random.randint(
            1, self.max_job_repetition + 1, self.n_jobs)

        # Initialize the job buffer with dummy operations
        self.job_buffer = np.full((self.n_jobs, self.max_job_repetition, Job.max_n_operations, 6), -1, dtype=np.int16)

        self.job_infos = [[] for _ in range(self.n_jobs)]

        for i in range(self.n_jobs):
            for j in range(self.job_repetition[i]):
                self.job_buffer[i, j] = self.jobs[i].encode()

                self.job_infos[i].append(copy.deepcopy(self.jobs[i]))

    def _get_valid_actions(self):
        valid_actions = np.zeros((self.n_machines, self.n_jobs), dtype=bool)
        cur_op_ids = []

        for n in range(self.n_jobs):
            if self.job_cursors[n] == -1:
                continue
            cur_op_ids.append(self.job_buffer[n, self.job_cursors[n], self.op_cursors[n], 0])

        for m in range(self.n_machines):
            for n in range(self.n_jobs):
                if self.job_cursors[n] == -1:  # job is finished
                    continue

                if cur_op_ids[n] == -1:  # dummy or finished
                    continue

                cur_op_queue = self.job_infos[n][self.job_cursors[n]].operation_queue
                cur_op = cur_op_queue[self.op_cursors[n]]

                # Check if the predecssor is finished later than the last time of the machine
                if cur_op.predecessor and cur_op_queue[self.op_cursors[n] - 1].end_time > self.machines[m].last_time:
                    continue

                if self.machines[m].can_process_op(cur_op.type):
                    valid_actions[m, n] = True

        return valid_actions

    def schedule_selected_job(self, machine_id: int, job_id: int):
        self.valid_actions = self._get_valid_actions()

        if not self.valid_actions[machine_id, job_id]:
            raise ValueError("Invalid action")

        selected_machine = self.machines[machine_id]
        selected_job = self.job_infos[job_id][self.job_cursors[job_id]]
        selected_op = selected_job.operation_queue[self.op_cursors[job_id]]

        # Update operation data
        selected_op.machine = selected_machine.id

        # Find the first available window for the operation
        # (bigger than the processing time of the operation)
        windows = []
        for i in range(len(selected_machine.scheduled_ops) - 1):
            windows.append((selected_machine.scheduled_ops[i].end_time,
                            selected_machine.scheduled_ops[i + 1].start_time))
        window_found = False
        for i in range(len(windows)):
            if windows[i][1] - windows[i][0] >= selected_op.processing_time:
                selected_op.start_time = windows[i][0]
                selected_op.end_time = selected_op.start_time + selected_op.processing_time
                window_found = True
                break

        # Update machine data
        selected_machine.up_time += selected_op.processing_time
        selected_machine.scheduled_ops.append(selected_op)

        if window_found:
            # Sort the operations in the machine by start time
            selected_machine.scheduled_ops = sorted(
                selected_machine.scheduled_ops, key=lambda x: x.start_time)
        else:
            # Push the operation to the end of the schedule
            selected_op.start_time = selected_machine.last_time
            selected_op.end_time = selected_op.start_time + selected_op.processing_time
            selected_machine.last_time = selected_op.end_time

        # Add the operation to the schedule table
        self.schedule_table[machine_id, self.schedule_table_indices[machine_id]] = selected_op.encode()
        self.schedule_table_indices[machine_id] += 1

        # Update all machines' utilization
        self.global_last_time = max(self.global_last_time, selected_machine.last_time)
        for machine in self.machines:
            if self.global_last_time == 0:
                print(window_found, selected_op.start_time, selected_op.end_time, selected_machine.scheduled_ops)
            machine.utilization = float(machine.up_time / self.global_last_time)

        # Update machine info
        self.machine_info = np.array([machine.encode() for machine in self.machines], dtype=np.float32)

        # Remove the operation from the job buffer
        self.job_buffer[job_id, self.job_cursors[job_id], self.op_cursors[job_id]].fill(-1)
        self.op_cursors[job_id] += 1

        if self.op_cursors[job_id] == Job.max_n_operations:
            self.job_cursors[job_id] += 1
            self.op_cursors[job_id] = 0

        if self.job_cursors[job_id] == self.max_job_repetition:
            self.job_cursors[job_id] = -1
            self.op_cursors[job_id] = -1

        # Update valid actions
        self.valid_actions = self._get_valid_actions()

    def is_done(self):
        # All jobs are finished
        # or there is no valid action
        return np.all(self.job_cursors == -1) or not np.any(self.valid_actions)

    def get_state(self):
        return {
            'schedule_table': self.schedule_table,
            'job_buffer': self.job_buffer,
            'machine_info': self.machine_info,
            'valid_actions': self.valid_actions
        }

    def get_info(self):
        return {
            'repetition': self.job_repetition,
            'makespan': self.get_make_span(),
            'utilization': [machine.utilization for machine in self.machines],
            'scheduled_ops': [machine.scheduled_ops for machine in self.machines],
            'job_infos': self.job_infos,
        }

    def get_mean_machine_utilization(self):
        return np.mean([machine.utilization for machine in self.machines])

    def get_make_span(self):
        return self.global_last_time

    def show_gantt_chart(self):
        if self.global_last_time == 0:
            # show an empty chart
            fig, ax = plt.subplots()
            ax.set_yticks([0])
            ax.set_yticklabels(['Machine'])
            ax.set_xticks([0])
            ax.set_xticklabels(['Time'])
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            plt.show()
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title('Gantt Chart of the Schedule')
        ax.set_yticks(range(self.n_machines))
        ax.set_yticklabels([f'Machine {i}\n ability:{self.machines[i].ability}' for i in range(self.n_machines)])
        ax.set_xticks(range(0, self.global_last_time + 1, 10))
        ax.set_xticklabels(range(0, self.global_last_time + 1, 10))
        ax.set_xlim(0, self.global_last_time)
        ax.set_ylim(-1, self.n_machines)

        for machine_id in range(self.n_machines):
            for op in self.machines[machine_id].scheduled_ops:
                op_block = mpatches.Rectangle(
                    (op.start_time, machine_id - 0.5), op.end_time - op.start_time, 1, facecolor=self.jobs[op.job_id].color, edgecolor='black', linewidth=1)
                ax.add_patch(op_block)

        # Add legend for job repetition
        legend_patches = []
        for i in range(self.n_jobs):
            legend_patches.append(mpatches.Patch(color=self.jobs[i].color, label=f'Job {i} x {self.job_repetition[i]}'))
        ax.legend(handles=legend_patches)

        plt.show()
