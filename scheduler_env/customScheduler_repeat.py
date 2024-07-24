import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from stable_baselines3.common.env_checker import check_env
import matplotlib.colors as mcolors
import seaborn as sns
import heapq

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

class JobInfo:
    def __init__(self, name, color, operations, index = None):
        self.name = name
        self.color = color
        self.operation_queue = [Operation(op_info, job_id= str(int(name[4:])-1), color=color, job_index = index) for op_info in operations]

class Job(JobInfo):
    def __init__(self, job_info, index, deadline):
        super().__init__(job_info['name'], job_info['color'], job_info['operations'], index)
        self.index = index
        self.deadline = deadline
        self.estimated_tardiness = 0
        self.time_exceeded = 0
        
    def __lt__(self, other):
        # Define the comparison first by estimated_tardiness descending and then by index ascending
        if self.estimated_tardiness == other.estimated_tardiness:
            return self.index < other.index
        return self.estimated_tardiness > other.estimated_tardiness
    
    def __str__(self) -> str:
        pass

class Operation():
    def __init__(self, operation_info, job_id, color, job_index = None):
        self.sequence = None  # 초기화 시점에는 설정되지 않음
        self.index = operation_info['index']
        self.type = type_encoding(operation_info['type'])
        self.duration = operation_info['duration']
        self.predecessor = operation_info['predecessor']
        self.earliest_start = operation_info['earliest_start'] if operation_info['earliest_start'] else 0
        self.start = None  # 초기화 시점에는 설정되지 않음
        self.finish = None  # 초기화 시점에는 설정되지 않음
        self.machine = -1
        self.job = job_id
        self.job_index = job_index
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
            'job': self.job,
            'job_index': self.job_index,
        }

    def __str__(self):
        return f"job : {self.job}, index : {self.index} | ({self.start}, {self.finish})"
    
class customRepeatableScheduler():
    def __init__(self, jobs, machines) -> None:
        self.machines = [Machine(machine_info)
                          for machine_info in machines]
        self.job_infos = [JobInfo(job_info["name"], job_info["color"], job_info["operations"]) for job_info in jobs]
        self.jobs = []
        for job_info in jobs:
            job_list = []
            for i, deadline in enumerate(job_info['deadline']):
                job = Job(job_info, i, deadline)
                job_list.append(job)
            heapq.heapify(job_list)
            self.jobs.append(job_list)

        self.operations = [operation for job in self.job_infos for operation in job.operation_queue]
        
        len_jobs = len(self.jobs)
        # Reset 할 때 DeepCopy를 위해 원본을 저장해둠
        self.original_jobs = copy.deepcopy(self.jobs)
        self.original_machines = copy.deepcopy(self.machines)
        self.original_operations = copy.deepcopy(
            [operation for job in self.job_infos for operation in job.operation_queue])
        
        self.schedule_buffer = [[-1, -1] for _ in range(len_jobs)]
        
        # 4 : 각 리소스별 operation 수 최댓값
        self.original_job_details = np.ones(
            (len_jobs, 4, 2), dtype=np.int8) * -1
        for j in range(len_jobs):
            for o in range(len(self.job_infos[j].operation_queue)):
                self.original_job_details[j][o][0] = int(
                    self.job_infos[j].operation_queue[o].duration // 100)
                self.original_job_details[j][o][1] = int(
                    self.job_infos[j].operation_queue[o].type)
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

        self.machine_operation_rate = np.zeros(
            len(self.machines), dtype=np.float32)

        self.job_term = 0
        self.machine_term = 0

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

        self.schedule_buffer = [[-1, -1] for _ in range(len(self.jobs))]


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

        self.machine_operation_rate = np.zeros(
            len(self.machines), dtype=np.float32)

        self.update_state(None)

        # 기록을 위한 변수들
        self.current_schedule = []
        self.num_scheduled_operations = 0
        self.num_steps = 0
        self.last_finish_time = 0
        self.valid_count = 0

        return self.get_observation(), self.get_info() 
    
    def action_masks(self):
        self.update_legal_actions()
        self.action_mask = self.legal_actions.flatten()
        return self.action_mask

    def update_state(self, action=None):
        if action is not None:
            self.valid_count += 1
            self._schedule_operation(action)
            self._update_job_state()
            self._update_schedule_buffer()
            self._update_machine_state()
            self.last_finish_time = self._get_final_operation_finish()
        else:
            self._update_schedule_buffer()
            self._update_machine_state(init=True)
            self._update_job_state()

    def update_legal_actions(self):
        # Initialize legal_actions
        self.legal_actions = np.ones(
            (len(self.machines), len(self.jobs)), dtype=bool)

        for job_index in range(len(self.jobs)):
            # 1. 선택된 Job의 모든 Operation가 이미 종료된 경우
            if self.schedule_buffer[job_index] == [-1, -1]:
                self.legal_actions[:, job_index] = False

        for machine_index in range(len(self.machines)):
            # 2. 선택된 Machine가 선택된 Job의 Operation의 Type을 처리할 수 없는 경우
            machine = self.machines[machine_index]
            for job_index in range(len(self.jobs)):
                operation_index = self.schedule_buffer[job_index][1]
                if operation_index == -1:
                    # Debug message to check if the operation_index is -1
                    # print(f"Job {job_index} has no operations to schedule.")
                    break
                operation = self.job_infos[job_index].operation_queue[operation_index]
                if not machine.can_process_operation(operation.type):
                    self.legal_actions[machine_index, job_index] = False

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

    def _update_job_state(self):
        for job_list in self.jobs:
            for job in job_list:
                remaining_operations = [op for op in job.operation_queue if op.finish is None]
                
                if not remaining_operations:
                    job.estimated_tardiness = -1
                    if job.deadline < job.operation_queue[-1].finish:
                        job.time_exceeded = job.operation_queue[-1].finish - job.deadline
                    continue
                
                earliest_operation = remaining_operations[0]
                earliest_machine_end_times = [
                    machine.operation_schedule[-1].finish if machine.operation_schedule else 0
                    for machine in self.machines if earliest_operation.type in machine.ability
                ]
                
                if earliest_machine_end_times:
                    earliest_machine_end_time = min(earliest_machine_end_times)
                else:
                    earliest_machine_end_time = 0

                estimated_finish_time = max(earliest_machine_end_time, earliest_operation.earliest_start) + earliest_operation.duration
                remaining_durations = [op.duration for op in remaining_operations[1:]]
                estimated_total_duration = sum(remaining_durations)
                
                job.estimated_tardiness = (estimated_finish_time + estimated_total_duration - job.deadline) / job.deadline

        # Rebuild the heap based on the updated estimated tardiness values
        for job_list in self.jobs:
            heapq.heapify(job_list)

    # 이거 맘에 안 듦
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

    def _update_schedule_buffer(self):
        # Clear the current schedule buffer

        for i in range(len(self.schedule_buffer)):
            if all(job.estimated_tardiness == -1 for job in self.jobs[i]):
                self.schedule_buffer[i] = [-1, -1]
                continue
            
            # Get the job with the highest estimated tardiness from the heap
            job = heapq.heappop(self.jobs[i])
            # Find the first unscheduled operation in the job
            for j, operation in enumerate(job.operation_queue):
                if operation.finish is None:
                    # Append the job index and operation index to the schedule buffer
                    self.schedule_buffer[i] = [job.index, j]
                    break
            
            # Push the job back into the heap
            heapq.heappush(self.jobs[i], job)

    def _schedule_operation(self, action):
        # Implement the scheduling logic based on the action
        # You need to update the start and finish times of the operations
        # based on the selected operation index (action) and the current state.

        # Example: updating start and finish times
        selected_machine = self.machines[action[0]]
        selected_job = self.jobs[action[1]][0]
        selected_operation = selected_job.operation_queue[self.schedule_buffer[action[1]][1]]
        #print(selected_operation)
        operation_earliest_start = selected_operation.earliest_start
        
        # Check for predecessor's finish time
        if selected_operation.predecessor is not None:
            predecessor_operation = next(
                op for op in selected_job.operation_queue if op.index == selected_operation.predecessor
            )
            operation_earliest_start = max(operation_earliest_start, predecessor_operation.finish)

        operation_duration = selected_operation.duration
        machine_operations = sorted(
            selected_machine.operation_schedule, key=lambda operation: operation.start
        )

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

        self.current_schedule.append(selected_operation)
        selected_machine.operation_schedule.append(selected_operation)
        self.num_scheduled_operations += 1
        return

    def get_observation(self):

        observation = {
            'action_mask': self.action_masks(),
            'job_details': self.current_job_details,
            'machine_operation_rate': self.machine_operation_rate,
            'machine_types': self.machine_types,
            'operation_schedules': self.operation_schedules,
            'schedule_buffer': [elem[1] for elem in self.schedule_buffer],
            'estimated_tardiness': [self.jobs[i][elem[0]].estimated_tardiness if elem[0] != -1 else -1 for i, elem in enumerate(self.schedule_buffer)]
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
            'schedule_buffer': self.schedule_buffer,
            'current_schedule': self.current_schedule,
            'job_deadline': [job.deadline for job_list in self.jobs for job in job_list],
            'job_time_exceeded': [job.time_exceeded for job_list in self.jobs for job in job_list]
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
        current_schedule = [operation.to_dict() for operation in self.current_schedule]

        scheduled_df = list(filter(lambda operation: operation['sequence'] is not None, current_schedule))
        scheduled_df = pd.DataFrame(scheduled_df)

        if scheduled_df.empty:
            plt.figure(figsize=(12, 6))
            plt.title("Operation Schedule Visualization")
            return plt

        fig, ax = plt.subplots(figsize=(12, 6))
        legend_jobs = set()

        for i in range(len(self.machines)):
            machine_operations = scheduled_df[scheduled_df['machine'] == i]
            line_offset = i - 0.9

            for index, operation in machine_operations.iterrows():
                base_color = mcolors.to_rgba(operation['color'])
                job_index = operation["job_index"]
                shade_factor = (job_index + 1) / (len(self.jobs[0]) + 1)
                shaded_color = tuple([min(max(shade * shade_factor, 0), 1) if idx < 3 else shade for idx, shade in enumerate(base_color)])

                job_label = f'Job {int(operation["job"]) + 1} - Repeat {job_index + 1}'
                if job_label not in legend_jobs:
                    legend_jobs.add(job_label)
                else:
                    job_label = None

                ax.bar(
                    x=operation["start"] + operation["duration"] / 2,
                    height=0.8,
                    width=operation["duration"],
                    bottom=line_offset,
                    color=shaded_color,
                    alpha=0.7,
                    label=job_label,
                )

        ax.set_yticks(np.arange(0, len(self.machines)))
        ax.set_yticklabels([machine.name for machine in self.machines])

        ax.set(ylabel="Machine", xlabel="Time")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f'Operation Schedule Visualization step {num_steps}')
        plt.rcParams['figure.max_open_warning'] = 0

        return fig

    def is_legal(self, action):
        return self.legal_actions[action[0], action[1]]

    def is_done(self):
        # 모든 Job의 Operation가 종료된 경우 Terminated를 True로 설정한다
        # 또한 legal_actions가 전부 False인 경우도 Terminated를 True로 설정한다
        return all([job.operation_queue[-1].finish is not None for job_list in self.jobs for job in job_list]) or not np.any(self.legal_actions)
    
    def _get_final_operation_finish(self):
        return max(self.current_schedule, key=lambda x: x.finish).finish

    def calculate_step_reward(self):
        self.machine_term = 0.0
        if np.any(self.machine_operation_rate):
            self.machine_term = np.mean(self.machine_operation_rate)
        return self.machine_term
    
        # Schedule Buffer에 올라온 Job 들의 Estimated Tardiness 평균에 -1을 곱한 것을 반환
        # Estimated Tardiness가 -1인 Job은 Schedule Buffer에 올라오지 않는다
        # return -np.mean([job.estimated_tardiness for job_list in self.jobs for job in job_list if job.estimated_tardiness != -1])

    def calculate_final_reward(self, weight_final_time = 80, weight_job_deadline = 20, weight_op_rate = 0, target_time = 1000):
        def final_time_to_reward(target_time):
            if target_time >= self._get_final_operation_finish():
                return 1
            # Final_time이 target_time에 비해 몇 퍼센트 초과되었는지를 바탕으로 100점 만점으로 환산하여 점수 계산
            return max(0, 1 - ((self._get_final_operation_finish() - target_time) / target_time))
        
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
            # 각 Machine의 Hole을 점수로 만든다
            # return sum([machine.operation_rate for machine in self.machines]) / len(self.machines)
        
        def job_deadline_to_reward():
            sum_of_late_rate = 0
            total_job_length = 0
            for job_list in self.jobs:
                total_job_length += len(job_list)
                for job in job_list:
                    if job.time_exceeded > 0:
                        sum_of_late_rate += (job.time_exceeded / job.deadline)

            sum_of_late_rate /= total_job_length
            return max(0, 1 - sum_of_late_rate)

        final_reward_by_op_rate = weight_op_rate * operation_rate_to_reward()
        final_reward_by_final_time = weight_final_time * final_time_to_reward(target_time) 
        final_reward_by_job_deadline = weight_job_deadline * job_deadline_to_reward()

        return final_reward_by_op_rate + final_reward_by_final_time + final_reward_by_job_deadline


    