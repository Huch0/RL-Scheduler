import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import numpy as np
import json
import os
import torch
from torch_geometric.data import Data


class GraphJSSPEnv(gym.Env):
    def __init__(self, instance_config=None):

        self.graph = DisjunctiveGraph(instance_config)

        # self.node_space = spaces.Box(low=0, high=10000, shape=(2,), dtype=np.int16)
        # self.observation_space = spaces.Dict({
        #     'valid_actions': spaces.MultiBinary(self.graph.n_nodes),
        #     'graph': spaces.Graph(node_space=self.node_space)
        # })

        self.num_steps = 0
        self.max_steps = 10_000
        self.valid_actions = []

    def reset(self, seed=0):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed)
        self.num_steps = 0

        self.graph.reset()

        return self._get_observation(), self._get_info()

    def step(self, action):
        """
        action: node index
        """
        self.num_steps += 1

        reward = 0

        if action not in self.valid_actions:
            reward = -10
        else:
            self.graph.schedule_selected_op(action)
            self._get_step_reward()

        terminated = self.graph.is_terminated()
        if terminated:
            reward += self._get_final_reward()

        truncated = self.num_steps >= self.max_steps

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_step_reward(self):
        pass

    def _get_final_reward(self):
        pass

    def render(self, mode='human'):
        pass

    def _get_observation(self):
        self.valid_actions = self.graph.get_valid_op_ids()
        return {
            'valid_actions': self.valid_actions,
            'graph': self.graph
        }

    def _get_info(self):
        pass

class DisjunctiveGraph:
    def __init__(self, instance_config=None):
        if instance_config is None:
            dir_path = os.path.join(os.path.dirname(__file__), '../instances')
            instance_config = {
                'type': 'standard',
                'path': os.path.join(dir_path, 'standard/ta01'),
                'repeat': [1] * 15
            }

        self.instance_config = instance_config
        
        self.feature_names = {
            'is_scheduled': 0,
            'completion_time_lower_bound': 1
        }

        self.num_jobs = 0
        self.num_machines = 0
        self.jobs = []
        self.machines = []
        # node id 0, 1 : dummy nodes s, t
        self.operations = [Operation(0, -1, -1, -1), Operation(1, -1, -1, -1)]

        if instance_config['type'] == 'standard':
            with open(instance_config['path'], 'r') as f:
                # First line contains the number of jobs and machines
                self.num_jobs, self.num_machines = map(int, f.readline().split())
                self.jobs = [Job(job_id) for job_id in range(self.num_jobs)]
                self.machines = [Machine(machine_id) for machine_id in range(self.num_machines)]

                op_id = 2

                # Following lines contain the operations for each job
                for job_id in range(self.num_jobs):
                    # Each line: machine_id processing_time machine_id processing_time ...
                    line = list(map(int, f.readline().split()))
                    for i in range(0, len(line), 2):
                        machine_id, processing_time = line[i], line[i+1]
                        self.operations.append((op_id, job_id, machine_id, processing_time))
                        self.jobs[job_id].operation_queue.append(op_id)

                        op_id += 1

        self.data = self.build_graph()

    def build_graph(self):
        # Node features (is_scheduled, completion_time_lower_bound)
        node_features = torch.zeros((len(self.operations), 2), dtype=torch.int32)
        edge_index = []

        # Dummy nodes
        node_features[0] = torch.tensor([1, 0], dtype=torch.int32)
        node_features[1] = torch.tensor([0, 0], dtype=torch.int32)

        # Conjunctive edges from s to first operation of each job
        edge_index.extend([(0, job.operation_queue[0]) for job in self.jobs])

        for job in self.jobs:
            Clb = 0
            for op_id in job.operation_queue:
                Clb += self.operations[op_id].processing_time
                self.operations[op_id].completion_time_lower_bound = Clb
                node_features[op_id] = torch.tensor([0, Clb], dtype=torch.int32)

                # Conjunctive edges between operations of the same job
                if op_id != job.operation_queue[-1]:
                    edge_index.append((op_id, job.operation_queue[job.operation_queue.index(op_id) + 1]))

        # Conjunctive edges from last operation of each job to t
        edge_index.extend([(job.operation_queue[-1], 1) for job in self.jobs])

        # Convert edge_index list of tuples to a PyTorch tensor of shape [2, num_edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create PyG graph with corrected edge_index
        graph = Data(x=node_features, edge_index=edge_index)

        return graph

    def get_valid_op_ids(self):
        """
        valid actions: first operations of each job that are not scheduled
        """
        valid_op_ids = []

        for job in self.jobs:
            for op_id in job.operation_queue:
                if self.operations[op_id].machine == -1:
                    valid_op_ids.append(op_id)
                    break

        return valid_op_ids

    def schedule_selected_op(self, op_id):
        selected_machine = self.selected_machine[self.operations[op_id].machine_id]
        selected_op = self.operations[op_id]

        # Find the first available time slot for the selected opeartion
        # (bigger than the processing time of the operation)
        # The start time should be later than the end time of the predecessor
        start_time = -1
        p = selected_op.processing_time
        Clb = selected_op.completion_time_lower_bound
        schedule = selected_machine.scheduled_ops
        d_edges = selected_machine.disjunctive_edges

        if schedule and schedule[0].start_time - Clb >= p:
            start_time = Clb

            # Add new edge from selected operation to the first operation in the schedule
            self.data.edge_index = torch.cat([self.data.edge_index, torch.tensor([[op_id], [schedule[0].id]])], dim=1)
            # Store the index of the new edge in edge_index
            d_edges.insert(0, self.data.edge_index.shape[1] - 1)

            # Push the operation to the front of the schedule
            schedule.insert(0, selected_op)

        else:
            for i in range(1, len(selected_machine.scheduled_ops)):
                if schedule[i].start_time - max(schedule[i-1].end_time, Clb) >= p:
                    start_time = max(schedule[i-1].end_time, Clb)

                    prev_edge_id = d_edges[i-1]
                    next_op_id = self.data.edge_index[1][prev_edge_id]
                    # Redirect the edge from the previous operation to the selected operation
                    self.data.edge_index[1][prev_edge_id] = op_id
                    # Add new edge from selected operation to the next operation in the schedule
                    self.data.edge_index = torch.cat(
                        [self.data.edge_index, torch.tensor([[op_id], [next_op_id]])], dim=1)
                    # Store the index of the new edge in edge_index
                    d_edges.insert(i, self.data.edge_index.shape[1] - 1)

                    schedule.insert(i, selected_op)
                    break

        if start_time == -1:
            start_time = max(schedule[-1].end_time, Clb)
            if len(schedule) != 0:
                # Add new edge from the last operation in the schedule to the selected operation
                self.data.edge_index = torch.cat(
                    [self.data.edge_index, torch.tensor([[schedule[-1].id], [op_id]])], dim=1)
                # Store the index of the new edge in edge_index
                d_edges.append(self.data.edge_index.shape[1] - 1)
            schedule.append(selected_op)

        # Update the start and end time of the selected operation
        selected_op.start_time = start_time
        selected_op.end_time = start_time + p
        selected_op.machine = selected_machine

        # Update the node features
        self.data.x[op_id] = torch.tensor([1, selected_op.end_time], dtype=torch.int32)

        # Update the utilization of the selected machine
        selected_machine.up_time += p
        selected_machine.last_time = schedule[-1].end_time
        selected_machine.utilization = selected_machine.up_time / selected_machine.last_time

        # Update the completion time lower bound of the following operations
        Clb = selected_op.end_time
        for oi in self.jobs[selected_op.job_id].operation_queue[self.jobs[selected_op.job_id].operation_queue.index(op_id) + 1:]:
            Clb += self.operations[oi].processing_time
            self.operations[oi].completion_time_lower_bound = Clb
            self.data.x[oi][self.feature_names['completion_time_lower_bound']] = Clb


class Job():
    def __init__(self, job_id):
        self.id = job_id
        self.operation_queue = []

    def __repr__(self):
        return f"Job {self.id}"


class Operation():
    def __init__(self, op_id, job_id, machine_id, processing_time):
        self.id = op_id
        self.job_id = job_id
        self.machine_id = machine_id
        self.processing_time = processing_time

        # Informations for runtime
        self.start_time = -1
        self.end_time = -1
        self.machine = -1

        # Node features
        self.is_scheduled = 0
        self.completion_time_lower_bound = 0

    def __repr__(self):
        return f"OP{self.id} - Job{self.job_id}, {self.type}, {self.processing_time}, {self.start_time}, {self.end_time}"


class Machine():
    def __init__(self, machine_id):
        self.id = machine_id

        # Informations for runtime
        self.utilization = float(0)
        self.last_time = 0
        self.up_time = 0

        self.scheduled_ops = []
        self.disjunctive_edges = []

    def __repr__(self):
        return f"Machine {self.id}, ability: {self.ability}"
