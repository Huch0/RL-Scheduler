import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import numpy as np
import json
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
import random

# Set the seed for Python's built-in random module
random.seed(0)

# Set the seed for NumPy's random module
np.random.seed(0)


class GraphJSSPEnv(gym.Env):
    def __init__(self, instance_config=None):

        self.graph = DisjunctiveGraph(instance_config)

        self.node_space = spaces.Box(low=0, high=10000, shape=(2,), dtype=np.int16)
        self.observation_space = spaces.Dict({
            'valid_actions': spaces.MultiBinary(self.graph.num_jobs),
            'graph': spaces.Graph(node_space=self.node_space, edge_space=None)
        })
        self.action_space = spaces.Discrete(self.graph.num_jobs, seed=0)

        self.num_steps = 0
        self.max_steps = 10_000
        self.valid_actions = []
        self.max_Clb_t = 0

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
            reward = self._get_step_reward()

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
        reward = self.max_Clb_t - self.graph.max_Clb_among_scheduled
        # reward = abs(reward) # for debugging
        self.max_Clb_t = self.graph.max_Clb_among_scheduled
        return reward

    def _get_final_reward(self):
        return 0

    def render(self, mode='human'):
        if self.max_Clb_t == 0:
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
        ax.set_yticks(range(self.graph.num_machines))
        ax.set_yticklabels([f'Machine {i}' for i in range(self.graph.num_machines)])

        if self.max_Clb_t > 10000:
            scale = 1000
        elif self.max_Clb_t > 1000:
            scale = 100
        else:
            scale = 10

        ax.set_xticks(range(0, self.max_Clb_t + 1, scale))
        ax.set_xticklabels(range(0, self.max_Clb_t + 1, scale))
        ax.set_xlim(0, self.max_Clb_t)
        ax.set_ylim(-1, self.graph.num_machines)

        for machine_id in range(self.graph.num_machines):
            for op in self.graph.machines[machine_id].scheduled_ops:
                op_block = mpatches.Rectangle(
                    (op.start_time, machine_id - 0.5), op.end_time - op.start_time, 1, facecolor=self.graph.jobs[op.job_id].color, edgecolor='black', linewidth=1)
                ax.add_patch(op_block)

        # Add legend for job repetition
        legend_patches = []
        for i in range(self.graph.num_jobs):
            legend_patches.append(mpatches.Patch(color=self.graph.jobs[i].color, label=f'Job {i}'))
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')

        plt.show()

    def _get_observation(self):
        self.valid_actions = self.graph.get_valid_job_ids()
        return {
            'valid_actions': self.valid_actions,
            'graph': self.graph
        }

    def _get_info(self):
        return {
            'num_steps': self.num_steps,
            'makespan': self.graph.max_Clb_among_scheduled
        }


class DisjunctiveGraph:
    def __init__(self, instance_config=None):
        if instance_config is None:
            dir_path = os.path.join(os.path.dirname(__file__), '../instances')
            instance_config = {
                'type': 'standard',
                'path': os.path.join(dir_path, 'standard/ft06.txt'),
                'repeat': [1] * 15
            }

        self.instance_config = instance_config

        self.feature_names = {
            'is_scheduled': 0,
            'completion_time_lower_bound': 1
        }

        self.num_jobs = 0
        self.num_machines = 0
        # node id 0, 1 : dummy nodes s, t
        self.original_operations = [Operation(0, -1, -1, -1), Operation(1, -1, -1, -1)]
        self.original_jobs = []
        self.original_machines = []

        if instance_config['type'] == 'standard':
            with open(instance_config['path'], 'r') as f:
                # First line contains the number of jobs and machines
                self.num_jobs, self.num_machines = map(int, f.readline().split())
                self.original_jobs = [Job(job_id) for job_id in range(self.num_jobs)]
                self.original_machines = [Machine(machine_id) for machine_id in range(self.num_machines)]

                op_id = 2

                # Following lines contain the operations for each job
                for job_id in range(self.num_jobs):
                    # Each line: machine_id processing_time machine_id processing_time ...
                    line = list(map(int, f.readline().split()))
                    for i in range(0, len(line), 2):
                        machine_id, processing_time = line[i], line[i+1]
                        self.original_operations.append(Operation(op_id, job_id, machine_id, processing_time))
                        self.original_jobs[job_id].operation_queue.append(op_id)

                        op_id += 1

        self.operations = deepcopy(self.original_operations)
        self.jobs = deepcopy(self.original_jobs)
        self.machines = deepcopy(self.original_machines)

        self.max_Clb_among_scheduled = 0

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

    def get_valid_job_ids(self):
        """
        valid actions: first operations of each job that are not scheduled
        """
        valid_job_ids = []

        for job in self.jobs:
            for op_id in job.operation_queue:
                if self.operations[op_id].machine == -1:
                    valid_job_ids.append(job.id)
                    break

        return valid_job_ids

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

    def schedule_selected_op(self, job_id):
        # Select the first operation of the selected job that is not scheduled
        op_id = -1
        for oi in self.jobs[job_id].operation_queue:
            if self.operations[oi].machine == -1:
                op_id = oi
                break
        selected_machine = self.machines[self.operations[op_id].machine_id]
        selected_op = self.operations[op_id]

        # Find the first available time slot for the selected opeartion
        # (bigger than the processing time of the operation)
        # The start time should be later than the end time of the predecessor
        start_time = -1
        p = selected_op.processing_time
        Clb = selected_op.completion_time_lower_bound
        earliest_start_time = Clb - p
        schedule = selected_machine.scheduled_ops
        d_edges = selected_machine.disjunctive_edges

        # Find the first available time slot at the beginning of the schedule
        if schedule and schedule[0].start_time - earliest_start_time >= p:
            start_time = earliest_start_time

            # Add new edge from selected operation to the first operation in the schedule
            self.data.edge_index = torch.cat([self.data.edge_index, torch.tensor([[op_id], [schedule[0].id]])], dim=1)
            # Store the index of the new edge in edge_index
            d_edges.insert(0, self.data.edge_index.shape[1] - 1)

            # Push the operation to the front of the schedule
            schedule.insert(0, selected_op)

        else:  # Find the first available time slot in the middle of the schedule
            for i in range(1, len(selected_machine.scheduled_ops)):
                if schedule[i].start_time - max(schedule[i-1].end_time, earliest_start_time) >= p:
                    start_time = max(schedule[i-1].end_time, earliest_start_time)

                    prev_edge_id = d_edges[i-1]
                    next_op_id = int(self.data.edge_index[1][prev_edge_id])
                    # Redirect the edge from the previous operation to the selected operation
                    self.data.edge_index[1][prev_edge_id] = op_id
                    # Add new edge from selected operation to the next operation in the schedule
                    self.data.edge_index = torch.cat(
                        [self.data.edge_index, torch.tensor([[op_id], [next_op_id]])], dim=1)
                    # Store the index of the new edge in edge_index
                    d_edges.insert(i, self.data.edge_index.shape[1] - 1)

                    schedule.insert(i, selected_op)
                    break

        # Put the selected operation at the end of the schedule
        if start_time == -1:
            if len(schedule) == 0:  # The schedule is empty
                start_time = earliest_start_time
                # No need to add new edge
            else:
                start_time = max(schedule[-1].end_time, earliest_start_time)
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
        selected_op.is_scheduled = 1
        selected_op.completion_time_lower_bound = selected_op.end_time

        # Update the maximum completion time lower bound among the scheduled operations
        self.max_Clb_among_scheduled = max(self.max_Clb_among_scheduled, selected_op.end_time)

        # Update the node features
        self.data.x[op_id] = torch.tensor([1, selected_op.end_time], dtype=torch.int32)

        # Update the utilization of the selected machine
        selected_machine.up_time += p
        selected_machine.last_time = schedule[-1].end_time
        selected_machine.utilization = selected_machine.up_time / selected_machine.last_time

        # Update the completion time lower bound of the following operations
        cur_Clb = selected_op.end_time
        for oi in self.jobs[selected_op.job_id].operation_queue[self.jobs[selected_op.job_id].operation_queue.index(op_id) + 1:]:
            cur_Clb += self.operations[oi].processing_time
            self.operations[oi].completion_time_lower_bound = cur_Clb
            self.data.x[oi][self.feature_names['completion_time_lower_bound']] = cur_Clb

    def reset(self):
        self.operations = deepcopy(self.original_operations)
        self.jobs = deepcopy(self.original_jobs)
        self.machines = deepcopy(self.original_machines)
        self.max_Clb_among_scheduled = 0

        self.data = self.build_graph()

    def is_terminated(self):
        return all([op.is_scheduled for op in self.operations[2:]])


    def visualize_graph(self):
        G = to_networkx(self.data)
        # Plot the graph
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_size=700, node_color="lightblue", font_weight="bold")
        plt.show()
class Job():
    def __init__(self, job_id):
        self.id = job_id
        self.operation_queue = []
        self.color = self.generate_color()

    def generate_color(self):
        r = random.randint(0, 255) / 255.0
        g = random.randint(0, 255) / 255.0
        b = random.randint(0, 255) / 255.0
        return (r, g, b)

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


if __name__ == "__main__":
    env = GraphJSSPEnv()
    # check_env(env, warn=True)

    step = 0
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        step += 1

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if reward != -10:
            print(action, reward, done, obs['valid_actions'])
            # obs['graph'].visualize_graph()
            # env.render()
        # env.render()

        if done:
            print("Goal reached!")
            print(step, info, total_reward)
            obs['graph'].visualize_graph()
            env.render()

    env.close()
