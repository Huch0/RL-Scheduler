import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from stable_baselines3.common.env_checker import check_env
# from stable_baselines3 import A2C, PPO, DQN
# from sb3_contrib import MaskablePPO
# from sb3_contrib.common.wrappers import ActionMasker

def type_encoding(type):
    type_code = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
                 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
    return type_code[type]

class Machine():
    def __init__(self, machines_dictionary):
        self.operation_schedule = []  # (operations)
        self.name = machines_dictionary['name']
        self.ability = self.ability_encoding(
            machines_dictionary['ability'])  # "A, B, C, ..."
        self.operation_rate = 0.0

    def __str__(self):
        # str_to_operations = [str(operation) for operation in self.operation_schedule]
        # return f"{self.name} : {str_to_operations}"
        return f"name : {self.name}, ability : {self.ability}"

    def ability_encoding(self, ability):
        return [type_encoding(type) for type in ability]

    def can_process_operation(self, operation_type):
        return operation_type in self.ability

class Job():
    def __init__(self, job_info, job_id):
        self.name = job_info['name']
        self.color = job_info['color']
        self.deadline = job_info['deadline']
        self.time_exceeded = 0
        self.density = 0
        self.operation_queue = [Operation(operation_info, job_id, self.color) for operation_info in job_info['operations']]

    def __str__(self):
        return f"{self.name}, {self.time_exceeded}/{self.deadline}"

class Operation():
    def __init__(self, operation_info, job_id, color):
        self.sequence = None  # 초기화 시점에는 설정되지 않음
        self.index = operation_info['index']
        self.type = type_encoding(operation_info['type'])
        self.duration = operation_info['duration']
        self.predecessor = operation_info['predecessor']
        self.earliest_start = operation_info['earliest_start']
        self.start = None  # 초기화 시점에는 설정되지 않음
        self.finish = None  # 초기화 시점에는 설정되지 않음
        self.machine = -1
        self.job = job_id
        self.color = color
        self.expected_start = 0
        self.estimated_tardiness = 0

    def to_dict(self):
        return {
            'sequence': self.sequence,
            'index': self.index,
            'type': self.type,
            'predecessor': self.predecessor,
            'earliest_start': self.earliest_start,
            'duration': self.duration,
            'start': self.start,
            'finish': self.finish,
            'machine': self.machine,
            'color': self.color,
            'job': self.job
        }

    def __str__(self):
        return f"job : {self.job}, index : {self.index} | ({self.start}, {self.finish})"
    
class customScheduler():
    def __init__(self, jobs, machines) -> None:
        self.machines = [Machine(machine_info)
                          for machine_info in machines]
        self.jobs = [Job(job_info, job_id) for job_id, job_info in enumerate(jobs)]
        self.operations = [operation for job in self.jobs for operation in job.operation_queue]
        
        len_machines = len(self.machines)
        len_jobs = len(self.jobs)
        # Reset 할 때 DeepCopy를 위해 원본을 저장해둠
        self.original_jobs = copy.deepcopy(self.jobs)
        self.original_machines = copy.deepcopy(self.machines)
        self.original_operations = copy.deepcopy(
            [operation for job in self.jobs for operation in job.operation_queue])
        self.num_operations = sum([len(job.operation_queue) for job in self.jobs])

        self.schedule_buffer = [-1 for _ in range(len(self.jobs))]
        self.current_operation_index = -1  

        # 4 : 각 리소스별 operation 수 최댓값
        self.original_job_details = np.ones(
            (len_jobs, 4, 2), dtype=np.int8) * -1
        for j in range(len_jobs):
            for o in range(len(self.jobs[j].operation_queue)):
                self.original_job_details[j][o][0] = int(
                    self.jobs[j].operation_queue[o].duration // 100)
                self.original_job_details[j][o][1] = int(
                    self.jobs[j].operation_queue[o].type)
        self.current_job_details = copy.deepcopy(self.original_job_details)

        self.job_state = None
        self.machine_types = None
        self.operation_schedules = None
        # self.action_space = spaces.MultiDiscrete([len_machines, len_jobs])
        
        self.action_mask = np.ones(
            shape=(len(self.machines) * len(self.jobs)), dtype=bool)
        self.legal_actions = np.ones(
            shape=(len(self.machines), len(self.jobs)), dtype=bool)

        self.current_schedule = []
        self.num_scheduled_operations = 0
        self.num_steps = 0
        self.last_finish_time = 0
        self.valid_count = 0

        self.job_density = np.zeros(len(self.jobs), dtype=np.float32)
        self.machine_operation_rate = np.zeros(
            len(self.machines), dtype=np.float32)

        self.job_term = 0
        self.machine_term = 0

        self.graph = None

    def build_graph(self):
        # 노드 특징: (is_scheduled, completion_time_lower_bound)
        node_features = torch.zeros((self.num_operations + 2, 2), dtype=torch.float32)
        edge_index = []

        # 더미 노드 추가
        node_features[0] = torch.tensor([-1, 0], dtype=torch.float32)  # 시작 노드
        node_features[1] = torch.tensor([-1, 0], dtype=torch.float32)  # 종료 노드

        # 작업 간의 순서를 정의하는 엣지 추가
        for job in self.jobs:
            #Clb = 0
            expected_tardiness = 0
            for op in job.operation_queue:
                op_id = op.index + 2  # 노드 인덱스는 2부터 시작
                            
                if op.predecessor:
                    op.expected_start = self.operations[op.predecessor].expected_start + self.operations[op.predecessor].duration
                else:
                    op.expected_start = int(op.earliest_start) if op.earliest_start else 0
                    expected_tardiness = (op.expected_start + sum([op.duration for op in job.operation_queue]) - job.deadline) / job.deadline

                node_features[op_id] = torch.tensor([-1, expected_tardiness], dtype=torch.float32)  # [할당된 머신, 추정 Tardiness]
                # 작업 간의 순서 엣지 추가
                if op.predecessor is not None:
                    edge_index.append((op.predecessor + 2, op_id))
                else:
                    edge_index.append((0, op_id))  # 시작 노드에서 첫 작업으로

            edge_index.append((job.operation_queue[-1].index + 2, 1))  # 마지막 작업에서 종료 노드로

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        graph = Data(x=node_features, edge_index=edge_index)

        return graph
    
    def update_graph(self, action):
        selected_machine_id = action[0]
        selected_machine = self.machines[selected_machine_id]
        selected_operation = self.operations[self.current_operation_index]
        
        # 노드 특징 업데이트
        node_id = self.current_operation_index + 2
        self.graph.x[node_id][0] = selected_machine_id  # 할당된 머신
 
        for job_index, op_index in enumerate(self.schedule_buffer):
            if op_index < 0:
                continue  # 작업이 남아 있지 않으면 건너뜀

            job = self.jobs[job_index]
            op = job.operation_queue[op_index]

            earliest_machine_end_time = min(
                [machine.operation_schedule[-1].finish if machine.operation_schedule else 0 for machine in self.machines if op.type in machine.ability]
            )
            
            early_start = 0
            if op.earliest_start:
                early_start = op.earliest_start
            op.expected_start = max(earliest_machine_end_time, early_start)
            estimated_finish_time = op.expected_start + op.duration
            remaining_durations_after_op = [remaining_op.duration for remaining_op in job.operation_queue[op_index+1:] if remaining_op.finish is None]
            expected_tardiness = (estimated_finish_time + sum(remaining_durations_after_op) - job.deadline) / job.deadline
            op_id = op.index + 2
            self.graph.x[op_id][1] = expected_tardiness  # 추정 Tardiness 업데이트

        # 엣지 추가: 이전 작업과 다음 작업 간의 순서 엣지 추가
        if selected_operation.predecessor is not None:
            self.graph.edge_index = torch.cat([self.graph.edge_index, torch.tensor([[node_id-1], [node_id]])], dim=1)

        # 다음 작업이 존재하는가? 그렇다면 아래 내용 수행
        if self.current_operation_index + 1 < len(self.operations) and self.operations[self.current_operation_index+1].predecessor:
            self.graph.edge_index = torch.cat([self.graph.edge_index, torch.tensor([[node_id], [node_id+1]])], dim=1)

    def visualize_graph(self):
        G = to_networkx(self.graph, node_attrs=['x'])
        pos = nx.spring_layout(G)

        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold')
        node_labels = {i: f"{i}\n({self.graph.x[i][0].item()}, {self.graph.x[i][1].item()})" for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        plt.show()

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # 환경과 관련된 변수들
        self.jobs = copy.deepcopy(self.original_jobs)
        self.machines = copy.deepcopy(self.original_machines)
        self.operations = copy.deepcopy(self.original_operations)
        self.current_job_details = copy.deepcopy(self.original_job_details)

        # 내부 동작을 위한 변수들
        # self.job_state 관한 추가설명 / Job 하나 당 가지는 정보는 아래와 같다
        # 1. 남은 operation 수
        # 2. 다음으로 수행할 Operation의 Type
        # 3. 다음으로 수행할 Operation의 earliest_start
        # 4. 다음으로 수행할 Operation의 duration
        # self.job_state = np.zeros((len(self.jobs), 4), dtype=np.int32)
        self.machine_types = np.zeros(
            (len(self.machines), 25), dtype=np.int8)
        self.operation_schedules = np.zeros(
            (len(self.machines), 50), dtype=np.int8)

        self.legal_actions = np.ones(
            (len(self.machines), len(self.jobs)), dtype=bool)
        self.action_mask = np.ones(
            (len(self.machines) * len(self.jobs)), dtype=bool)

        self.job_density = np.zeros(len(self.jobs), dtype=np.float32)
        self.machine_operation_rate = np.zeros(
            len(self.machines), dtype=np.float32)

        self.update_state(None)

        # 기록을 위한 변수들
        self.current_schedule = []
        self.num_scheduled_operations = 0
        self.num_steps = 0
        self.last_finish_time = 0
        self.valid_count = 0

        self.current_operation_index = -1

        return self.get_observation(), self.get_info() 
    
    def action_masks(self):
        self.update_legal_actions()
        self.action_mask = self.legal_actions.flatten()
        return self.action_mask

    def update_state(self, action=None):
        if action is not None:
            self.valid_count += 1
            self._schedule_operation(action)
            
            self.current_operation_index = self.jobs[action[1]].operation_queue[self.schedule_buffer[action[1]]].index  # 현재 작업의 index 업데이트

            self._update_schedule_buffer(action[1])
            # self._update_job_state(action)
            self._update_job_details(action[1])
            self._update_machine_state()
            self.last_finish_time = self._get_final_operation_finish()
            self.update_graph(action)           
        else:
            self._update_schedule_buffer(None)
            # self._update_job_state(None)
            self._update_machine_state(init=True)
            self.graph = self.build_graph()

    def update_legal_actions(self):
        # Initialize legal_actions
        self.legal_actions = np.ones(
            (len(self.machines), len(self.jobs)), dtype=bool)

        for job_index in range(len(self.jobs)):
            # 1. 선택된 Job의 모든 Operation가 이미 종료된 경우
            if self.schedule_buffer[job_index] < 0:
                self.legal_actions[:, job_index] = False

        for machine_index in range(len(self.machines)):
            # 2. 선택된 Machine가 선택된 Job의 Operation의 Type을 처리할 수 없는 경우
            machine = self.machines[machine_index]
            for job_index in range(len(self.jobs)):
                job = self.jobs[job_index]
                operation = job.operation_queue[self.schedule_buffer[job_index]]
                if not machine.can_process_operation(operation.type):
                    self.legal_actions[machine_index, job_index] = False

    def _update_job_details(self, job_index):
        selected_job = self.jobs[job_index]

        sum_operation_duration = 0
        performed_operations = []
        for t, operation in enumerate(selected_job.operation_queue):
            if operation.finish is not None:  # operation is already scheduled
                self.current_job_details[job_index][t][0] = -1
                self.current_job_details[job_index][t][1] = -1
                sum_operation_duration += operation.duration
                performed_operations.append(operation)

        if len(performed_operations) > 1:
            job_duration = performed_operations[-1].finish - \
                performed_operations[0].start
            selected_job.density = sum_operation_duration / job_duration
            self.job_density[job_index] = selected_job.density

        if len(performed_operations) == len(selected_job.operation_queue):
            if selected_job.deadline < selected_job.operation_queue[-1].finish:
                selected_job.time_exceeded = selected_job.operation_queue[-1].finish - selected_job.deadline

    def _update_machine_state(self, init=False):
        if init:
            for i, machine in enumerate(self.machines):
                self.machine_types[i] = [
                    1 if i in machine.ability else 0 for i in range(25)]
            return

        for r, machine in enumerate(self.machines):
            operation_schedule = machine.operation_schedule
            self.operation_schedules[r] = self._schedule_to_array(
                operation_schedule)

            # 선택된 리소스의 스케줄링된 Operation들
            if machine.operation_schedule:
                operation_time = sum(
                    [operation.duration for operation in machine.operation_schedule])

                machine.operation_rate = operation_time / self._get_final_operation_finish()
                self.machine_operation_rate[r] = machine.operation_rate

    def _schedule_to_array(self, operation_schedule):
        idle_time = []

        for operation in operation_schedule:
            idle_time.append((operation.start // 100, operation.finish // 100))

        def is_in_idle_time(time):
            for interval in idle_time:
                if interval[0] <= time < interval[1]:
                    return True
            return False

        result = []

        for i in range(50):
            result.append(is_in_idle_time(i))

        return result

    def _update_schedule_buffer(self, target_job=None):
        # target_job은 매번 모든 Job를 보는 계산량을 줄이기 위해 설정할 변수
        # None은 최초의 호출에서, 또는 Reset이 이뤄질 경우를 위해 존재
        if target_job == None:
            buffer_index = 0

            for job in self.jobs:
                # Assume job['steps'] is a list of operations for the current job

                selected_operation_index = -1

                for i in range(len(job.operation_queue)):
                    # 아직 스케줄링을 시작하지 않은 Operation를 찾는다
                    if job.operation_queue[i].finish is None:
                        selected_operation_index = i
                        break
                # 스케줄링 하지 않은 Operation를 발견했다면
                if selected_operation_index >= 0:
                    selected_operation = job.operation_queue[selected_operation_index]

                    # 만약 초기 시작 제한이 없다면
                    # 초기 시작 제한을 이전 Operation의 Finish Time으로 걸어주고 버퍼에 등록한다.
                    if selected_operation.earliest_start is None:
                        if selected_operation_index > 0:
                            selected_operation.earliest_start = job.operation_queue[selected_operation_index-1].finish

                self.schedule_buffer[buffer_index] = selected_operation_index
                buffer_index += 1

        # Action으로 인해 봐야할 버퍼의 인덱스가 정해짐
        else:
            selected_operation_index = -1
            job = self.jobs[target_job]
            for i in range(len(job.operation_queue)):
                # 아직 스케줄링을 시작하지 않은 Operation를 찾는다
                if job.operation_queue[i].finish is None:
                    selected_operation_index = i
                    break
            if selected_operation_index >= 0:
                selected_operation = job.operation_queue[selected_operation_index]
                if selected_operation.earliest_start is None:
                    if selected_operation_index > 0:
                        selected_operation.earliest_start = job.operation_queue[selected_operation_index-1].finish

            self.schedule_buffer[target_job] = selected_operation_index

    def _schedule_operation(self, action):
        # Implement the scheduling logic based on the action
        # You need to update the start and finish times of the operations
        # based on the selected operation index (action) and the current state.

        # Example: updating start and finish times
        selected_machine = self.machines[action[0]]
        selected_job = self.jobs[action[1]]
        selected_operation = selected_job.operation_queue[self.schedule_buffer[action[1]]]
        operation_earliest_start = selected_operation.earliest_start
        operation_index = selected_operation.index
        operation_duration = selected_operation.duration
        machine_operations = sorted(
            selected_machine.operation_schedule, key=lambda operation: operation.start)

        open_windows = []
        start_window = 0
        last_alloc = 0

        for scheduled_operation in machine_operations:
            machine_init = scheduled_operation.start

            if machine_init > start_window:
                open_windows.append([start_window, machine_init])
            start_window = scheduled_operation.finish

            last_alloc = max(last_alloc, start_window)

        # Fit the operation within the first possible window
        window_found = False
        if operation_earliest_start is None:
            operation_earliest_start = 0

        for window in open_windows:
            # Operation could start before the open window closes
            if operation_earliest_start <= window[1]:
                # Now let's see if it fits there
                potential_start = max(operation_earliest_start, window[0])
                if potential_start + operation_duration <= window[1]:
                    # Operation fits into the window
                    min_earliest_start = potential_start
                    window_found = True
                    break

        # If no window was found, schedule it after the end of the last operation on the machine
        if not window_found:
            if operation_earliest_start > 0:
                min_earliest_start = max(operation_earliest_start, last_alloc)
            else:
                min_earliest_start = last_alloc

        # schedule it
        selected_operation.sequence = self.num_scheduled_operations + 1
        selected_operation.start = min_earliest_start
        selected_operation.finish = min_earliest_start + operation_duration
        selected_operation.machine = action[0]

        # 사실 여기서 color랑 job를 주는건 적절치 않은 코드임!!!!
        selected_operation.color = self.jobs[action[1]].color
        selected_operation.job = action[1]

        self.current_schedule.append(selected_operation)
        selected_machine.operation_schedule.append(selected_operation)
        self.num_scheduled_operations += 1
        return

    def get_observation(self):
        max_edges = 100
        padded_edge_index = torch.zeros((2, max_edges), dtype=torch.long)
        num_edges = min(self.graph.edge_index.shape[1], max_edges)
        padded_edge_index[:, :num_edges] = self.graph.edge_index[:, :num_edges]

        observation = {
            'action_mask': self.action_masks(),
            'job_details': self.current_job_details,
            'job_density': self.job_density,
            'machine_operation_rate': self.machine_operation_rate,
            'num_operation_per_machine': np.array([len(machine.operation_schedule) for machine in self.machines]),
            'machine_types': self.machine_types,
            'operation_schedules': self.operation_schedules,
            # 'node_space': self.graph.x.numpy(),
            # 'edge_index': padded_edge_index.numpy()
        }

        return observation

    def get_info(self):
        return {
            'finish_time': self.last_finish_time,
            'legal_actions': self.legal_actions,
            'action_mask': self.action_mask,
            'job_details': self.current_job_details,
            'machine_score': self.machine_term,
            'machine_operation_rate': [machine.operation_rate for machine in self.machines],
            'job_density': [job.density for job in self.jobs],
            'schedule_buffer': self.schedule_buffer,
            'current_schedule': self.current_schedule,
            'job_deadline' : [job.deadline for job in self.jobs],
            'job_time_exceeded' : [job.time_exceeded for job in self.jobs]
        }

    def render(self, mode="seaborn", num_steps = 0):
        if mode == "console":
            # You can implement console rendering if needed
            pass
        elif mode == "seaborn":
            return self._render_seaborn(num_steps)
        elif mode == "rgb_array":
            return self._render_rgb_array(num_steps)

    def _render_seaborn(self, num_steps):
        fig = self._make_chart(num_steps)
        plt.show()

    def _render_rgb_array(self, num_steps):
        # Render the figure as an image
        fig = self._make_chart(num_steps)
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()

        # Convert the image to RGB array
        buf = canvas.buffer_rgba()
        width, height = canvas.get_width_height()
        rgb_array = np.frombuffer(
            buf, dtype=np.uint8).reshape((height, width, 4))

        return rgb_array

    def _make_chart(self, num_steps):
        # Create a DataFrame to store operation scheduling information
        current_schedule = [operation.to_dict() for operation in self.current_schedule]

        scheduled_df = list(
            filter(lambda operation: operation['sequence'] is not None, current_schedule))
        scheduled_df = pd.DataFrame(scheduled_df)

        if scheduled_df.empty:
            # Create an empty chart
            plt.figure(figsize=(12, 6))
            plt.title("Operation Schedule Visualization")
            return plt

        # Create a bar plot using matplotlib directly
        fig, ax = plt.subplots(figsize=(12, 6))
        legend_jobs = set()  # Set to store jobs already included in the legend

        for i in range(len(self.machines)):
            machine_operations = scheduled_df[scheduled_df['machine'] == i]

            # Discriminate rows by lines
            line_offset = i - 0.9  # Adjust the line offset for better visibility

            for index, operation in machine_operations.iterrows():
                job_label = f'Job {int(operation["job"]) + 1}'
                if job_label not in legend_jobs:
                    legend_jobs.add(job_label)
                else:
                    job_label = None

                ax.bar(
                    # Adjust 'x' to start from 'start'
                    x=operation["start"] + operation["duration"] / 2,
                    height=0.8,  # Height of the bar
                    width=operation["duration"],  # Width of the bar
                    bottom=line_offset,  # Discriminate rows by lines
                    color=operation['color'],
                    alpha=0.7,  # Transparency
                    label=job_label,  # Label for the legend
                )

        # Set y-axis ticks to show every machine
        ax.set_yticks(np.arange(0, len(self.machines)))
        ax.set_yticklabels(self.machines)

        ax.set(ylabel="Machine", xlabel="Time")
        # Place the legend outside the plot area
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f'Operation Schedule Visualization step {num_steps}')
        # 경고 무시 설정
        plt.rcParams['figure.max_open_warning'] = 0

        return fig

    def is_legal(self, action):
        return self.legal_actions[action[0], action[1]]

    def is_done(self):
        # 모든 Job의 Operation가 종료된 경우 Terminated를 True로 설정한다
        # 또한 legal_actions가 전부 False인 경우도 Terminated를 True로 설정한다
        return all([job.operation_queue[-1].finish is not None for job in self.jobs]) or not np.any(self.legal_actions)
    
    def _get_final_operation_finish(self):
        return max(self.current_schedule, key=lambda x: x.finish).finish

    def calculate_step_reward(self):
        self.machine_term = 0.0
        if np.any(self.machine_operation_rate):
            self.machine_term = np.mean(self.machine_operation_rate)
        return self.machine_term

    def calculate_final_reward(self, weight_final_time = 80, weight_job_deadline = 20, weight_op_rate = 0, target_time = 1000):
        def final_time_to_reward(target_time):
            if target_time >= self._get_final_operation_finish():
                return 1
            # Final_time이 target_time에 비해 몇 퍼센트 초과되었는지를 바탕으로 100점 만점으로 환산하여 점수 계산
            return max(0, 1 - (abs(target_time - self._get_final_operation_finish()) / target_time))
        
        #def operation_rate_to_reward(operation_rates, target_rate=1.0, penalty_factor=2.0):
            #total_reward = 0
            # for rate in operation_rates:
            #     # operation rate가 목표에 가까울수록 보상을 증가시킴
            #     reward = 2/(abs(rate - target_rate) - 2) + 2
            #     # operation rate의 차이에 따라 패널티를 부여함
            #     penalty = penalty_factor * abs(rate - target_rate)
            #     # 보상에서 패널티를 빼서 최종 보상을 계산함
            #     total_reward += reward - penalty
        def operation_rate_to_reward():
            return min([machine.operation_rate for machine in self.machines])
        
        def job_deadline_to_reward():
            sum_of_late_rate = 0
            for job in self.jobs:
                if job.time_exceeded > 0:
                    sum_of_late_rate += (job.time_exceeded / job.deadline)
            #sum_of_late_rate /= len(self.jobs)
            return max(0, 1 - sum_of_late_rate)

        final_reward_by_op_rate = weight_op_rate * operation_rate_to_reward()
        final_reward_by_final_time = weight_final_time * final_time_to_reward(target_time) 
        final_reward_by_job_deadline = weight_job_deadline * job_deadline_to_reward()

        return final_reward_by_op_rate + final_reward_by_final_time + final_reward_by_job_deadline


    