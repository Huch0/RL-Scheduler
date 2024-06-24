import gymnasium as gym
from gymnasium import spaces
import json
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
        return f"{self.name}"

    def ability_encoding(self, ability):
        return [type_encoding(type) for type in ability]

    def can_process_operation(self, operation_type):
        return operation_type in self.ability


class Job():
    def __init__(self, job_info):
        self.name = job_info['name']
        self.color = job_info['color']
        self.operation_queue = [Operation(operation_info)
                           for operation_info in job_info['operations']]
        self.density = 0

        # changed : add deadline, time_exceeded
        self.deadline = job_info['deadline']
        self.time_exceeded = 0

    def __str__(self):
        return f"{self.name}, {self.time_exceeded}/{self.deadline}"


class Operation():
    def __init__(self, operation_info):
        # self.sequence = operation_info['sequence']
        # self.index = operation_info['index']
        # Informations for rendering
        self.color = ""
        # Informations for runtime
        self.start = operation_info['start']
        self.finish = operation_info['finish']
        self.machine = -1
        self.job = -1
        self.type = type_encoding(operation_info['type'])
        self.predecessor = operation_info['predecessor']
        self.earliest_start = operation_info['earliest_start']
        self.duration = operation_info['duration']

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
        return f"job : {self.job}, step : {self.index} | ({self.start}, {self.finish})"


class SchedulingEnv(gym.Env):
    def _load_machines(self, file_path):
        machines = []

        with open(file_path, 'r') as file:
            data = json.load(file)

        for machine_data in data["machines"]:
            machine = {}
            machine['name'] = machine_data["name"]
            machine['ability'] = machine_data["type"].split(', ')
            machines.append(machine)

        return machines

    def _load_jobs(self, file):
        # Just in case we are reloading operations

        jobs = []  # 리턴할 용도
        jobs_new_version = []  # 파일 읽고 저장할 때 쓰는 용도
        f = open(file)

        # returns JSON object as  a dictionary
        data = json.load(f)
        f.close()
        jobs_new_version = data['jobs']

        for job in jobs_new_version:
            job_info = {}
            # Initial index of steps within job
            job_info['name'] = job['name']
            job_info['color'] = job['color']
            # changed : add deadline
            job_info['deadline'] = job['deadline']
            earliestStart = job['earliest_start']

            operations = []
            for operation in job['operations']:
                predecessor = operation['predecessor']
                operation_info = {}
                # Sequence is the scheduling job, the series of which defines a State or Node.
                operation_info['sequence'] = None
                operation_info['index'] = operation['index']
                operation_info['type'] = operation['type']
                if predecessor is None:
                    operation_info['predecessor'] = None
                    operation_info['earliest_start'] = earliestStart
                else:
                    operation_info['predecessor'] = predecessor
                    operation_info['earliest_start'] = None
                operation_info['duration'] = operation['duration']
                operation_info['start'] = None
                operation_info['finish'] = None

                operations.append(operation_info)

            job_info['operations'] = operations
            jobs.append(job_info)

        return jobs

    def __init__(self, machines="instances/Machines/v0-8.json", jobs="instances/Jobs/v0-12-deadline.json", render_mode="seaborn", weight_final_time=80, weight_job_deadline=20, weight_op_rate=0):
        super(SchedulingEnv, self).__init__()

        self.weight_final_time = weight_final_time
        self.weight_job_deadline = weight_job_deadline
        self.weight_op_rate = weight_op_rate
        machines = self._load_machines(machines)
        jobs = self._load_jobs(jobs)
        self.machines = [Machine(machine_info)
                          for machine_info in machines]
        self.jobs = [Job(job_info) for job_info in jobs]
        len_machines = len(self.machines)
        len_jobs = len(self.jobs)
        # Reset 할 때 DeepCopy를 위해 원본을 저장해둠
        self.original_jobs = copy.deepcopy(self.jobs)
        self.original_machines = copy.deepcopy(self.machines)
        self.original_operations = copy.deepcopy(
            [job.operation_queue for job in self.jobs])
        self.num_operations = sum([len(job.operation_queue) for job in self.jobs])

        self.schedule_buffer = [-1 for _ in range(len(self.jobs))]
        # 4 : 각 리소스별 operation 수 최댓값
        self.original_job_details = np.ones(
            (len_jobs, 4, 2), dtype=np.int8) * -1
        for o in range(len_jobs):
            for t in range(len(self.jobs[o].operation_queue)):
                self.original_job_details[o][t][0] = int(
                    self.jobs[o].operation_queue[t].duration // 100)
                self.original_job_details[o][t][1] = int(
                    self.jobs[o].operation_queue[t].type)
        self.current_job_details = copy.deepcopy(self.original_job_details)

        self.job_state = None
        self.machine_types = None
        self.operation_schedules = None
        # self.action_space = spaces.MultiDiscrete([len_machines, len_jobs])
        self.action_space = spaces.Discrete(len_machines * len_jobs)
        self.action_mask = np.ones(
            shape=(len(self.machines) * len(self.jobs)), dtype=bool)
        self.legal_actions = np.ones(
            shape=(len(self.machines), len(self.jobs)), dtype=bool)

        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(low=0, high=1, shape=(len_machines * len_jobs, ), dtype=np.int8),
            "job_details": spaces.Box(low=-1, high=25, shape=(len_jobs, 4, 2), dtype=np.int8),
            'job_density': spaces.Box(low=0, high=1, shape=(len_jobs, ), dtype=np.float32),
            'machine_operation_rate': spaces.Box(low=0, high=1, shape=(len_machines, ), dtype=np.float32),
            "num_operation_per_machine": spaces.Box(low=0, high=100, shape=(len_machines, ), dtype=np.int64),
            "machine_types": spaces.Box(low=0, high=1, shape=(len_machines, 25), dtype=np.int8),
            "operation_schedules": spaces.Box(low=0, high=1, shape=(len_machines, 50), dtype=np.int8)
        })

        self.current_schedule = []
        self.num_scheduled_operations = 0
        self.num_steps = 0
        self.invalid_count = 0
        self.last_finish_time = 0
        self.valid_count = 0

        self.job_density = np.zeros(len(self.jobs), dtype=np.float32)
        self.machine_operation_rate = np.zeros(
            len(self.machines), dtype=np.float32)

        self.job_term = 0
        self.machine_term = 0
        self.o = 2
        self.r = 3

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)

        # 환경과 관련된 변수들
        self.jobs = copy.deepcopy(self.original_jobs)
        self.machines = copy.deepcopy(self.original_machines)

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

        self._update_state(None)

        # 기록을 위한 변수들
        self.current_schedule = []
        self.num_scheduled_operations = 0
        self.num_steps = 0
        self.invalid_count = 0
        self.last_finish_time = 0
        self.valid_count = 0

        return self._get_observation(), self._get_info()  # empty info dict

    def step(self, action):
        # if action[0] < 0 or action[1] < 0 or action[0] >= len(self.machines) or action[1] >= len(self.jobs):
        #     raise ValueError(
        #         f"Received invalid action={action} which is not part of the action space"
        #     )
        if action < 0 or action >= len(self.machines) * len(self.jobs):
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )

        # Map the action to the corresponding machine and job
        selected_machine = action // len(self.jobs)
        selected_job = action % len(self.jobs)
        action = [selected_machine, selected_job]

        # error_action이 아니라면 step의 수를 증가시킨다
        self.num_steps += 1
        self._update_legal_actions()
        reward = 0

        if self.legal_actions[action[0]][action[1]]:
            self._update_state(action)
            reward = self._calculate_step_reward()
        else:  # Illegal action
            self.invalid_count += 1
            reward = -0.5

        # 모든 Job의 Operation가 종료된 경우 Terminated를 True로 설정한다
        # 또한 legal_actions가 전부 False인 경우도 Terminated를 True로 설정한다
        terminated = all([job.operation_queue[-1].finish is not None for job in self.jobs]
                         ) or not np.any(self.legal_actions)

        if terminated:
            reward += self._calculate_final_reward(self.weight_final_time, self.weight_job_deadline, self.weight_op_rate)
            # print(f"reward : {reward}")

        # reward += sum([operation.duration for operation in self.current_schedule]) / self._get_final_operation_finish()
        # 무한 루프를 방지하기 위한 조건
        truncated = bool(self.num_steps == 10000)
        if truncated:
            reward = -100

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _get_info(self):
        return {
            'finish_time': self.last_finish_time,
            'legal_actions': self.legal_actions,
            'action_mask': self.action_mask,
            'job_details': self.current_job_details,
            'job_score': self.job_term * self.o / (self.o + self.r),
            'machine_score': self.machine_term,
            'invalid_count': self.invalid_count,
            'machine_operation_rate': [machine.operation_rate for machine in self.machines],
            'job_density': [job.density for job in self.jobs],
            'schedule_buffer': self.schedule_buffer,
            'current_schedule': self.current_schedule,
            'job_deadline' : [job.deadline for job in self.jobs],
            'job_time_exceeded' : [job.time_exceeded for job in self.jobs]
        }

    def action_masks(self):
        self._update_legal_actions()
        self.action_mask = self.legal_actions.flatten()
        return self.action_mask

    def _update_state(self, action=None):
        if action is not None:
            self.valid_count += 1
            self._schedule_operation(action)
            self._update_schedule_buffer(action[1])
            # self._update_job_state(action)
            self._update_job_details(action[1])
            self._update_machine_state()
            self.last_finish_time = self._get_final_operation_finish()
        else:
            self._update_schedule_buffer(None)
            # self._update_job_state(None)
            self._update_machine_state(init=True)

    def _update_legal_actions(self):
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

    def _get_final_operation_finish(self):
        return max(self.current_schedule, key=lambda x: x.finish).finish

    def _calculate_final_reward(self, weight_final_time = 80, weight_job_deadline = 20, weight_op_rate = 0):
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
        final_reward_by_final_time = weight_final_time * final_time_to_reward(1000) 
        final_reward_by_job_deadline = weight_job_deadline * job_deadline_to_reward()

        return final_reward_by_op_rate + final_reward_by_final_time + final_reward_by_job_deadline
    
    def _calculate_step_reward(self):
        # self.job_term = 0
        self.machine_term = 0
        # check if all the elements are not 0
        # if np.any(self.job_density):
        #     self.job_term = 1 - self._gini_coefficient(self.job_density)
        if np.any(self.machine_operation_rate):
            # self.machine_term = 1 - \
            #     self._gini_coefficient(self.machine_operation_rate)
            self.machine_term = np.mean(self.machine_operation_rate)

        # return (self.o * self.job_term + self.r * self.machine_term) / (self.o + self.r)
        return self.machine_term

    @staticmethod
    def _gini_coefficient(array):
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    def _get_observation(self):
        observation = {
            'action_mask': self.action_masks(),
            'job_details': self.current_job_details,
            'job_density': self.job_density,
            'machine_operation_rate': self.machine_operation_rate,
            'num_operation_per_machine': np.array([len(machine.operation_schedule) for machine in self.machines]),
            'machine_types': self.machine_types,
            'operation_schedules': self.operation_schedules
        }

        return observation

    def render(self, mode="seaborn"):
        if mode == "console":
            # You can implement console rendering if needed
            pass
        elif mode == "seaborn":
            return self._render_seaborn()
        elif mode == "rgb_array":
            return self._render_rgb_array()

    def _render_seaborn(self):
        fig = self._make_chart()
        plt.show()

    def _render_rgb_array(self):
        # Render the figure as an image
        fig = self._make_chart()
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()

        # Convert the image to RGB array
        buf = canvas.buffer_rgba()
        width, height = canvas.get_width_height()
        rgb_array = np.frombuffer(
            buf, dtype=np.uint8).reshape((height, width, 4))

        return rgb_array

    def _make_chart(self):
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
        plt.title(f'Operation Schedule Visualization step {self.num_steps}')
        # 경고 무시 설정
        plt.rcParams['figure.max_open_warning'] = 0

        return fig

    def close(self):
        pass

if __name__ == "__main__":
    env = SchedulingEnv(machines="instances/Machines/v0-8.json",
                        jobs="instances/Jobs/v0-12-deadline.json")

    check_env(env)

    step = 0
    obs, _ = env.reset()

    while True:
        step += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("Goal reached!", "reward=", reward)
            env.render()
            break
