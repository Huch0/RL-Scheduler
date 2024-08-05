import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from stable_baselines3.common.env_checker import check_env
import matplotlib.colors as mcolors
import seaborn as sns
import heapq
from PIL import Image
import io

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
    
    def cal_best_finish_time(self, op_duration, op_type, op_earliest_start):
        if not self.can_process_operation(op_type):
            return -1
        
        if not self.operation_schedule:
            return op_earliest_start + op_duration
        
        operation_schedule = self.operation_schedule[::]
        operation_schedule.sort(key=lambda x: x.start)

        for i in range(len(operation_schedule)):
            if i == 0:
                if operation_schedule[i].start >= op_earliest_start + op_duration:
                    return op_earliest_start + op_duration
                else:
                    return operation_schedule[i].finish + op_earliest_start + op_duration 
            else:
                if op_earliest_start <= operation_schedule[i-1].finish:
                    if operation_schedule[i].start - operation_schedule[i-1].finish >= op_duration:
                        return operation_schedule[i-1].finish + op_duration
                else:
                    if operation_schedule[i].start - op_earliest_start >= op_duration:
                        return op_earliest_start + op_duration
            return max(operation_schedule[-1].finish, op_earliest_start) + op_duration
            

class JobInfo:
    def __init__(self, name, color, operations, index = None):
        self.name = name
        self.color = color
        self.operation_queue = [Operation(op_info, job_id= str(int(name[4:])-1), color=color, job_index = index) for op_info in operations]
        self.total_duration = sum([op.duration for op in self.operation_queue])

class Job(JobInfo):
    def __init__(self, job_info, index, deadline):
        super().__init__(job_info['name'], job_info['color'], job_info['operations'], index)
        self.index = index
        self.deadline = deadline
        self.estimated_tardiness = 0
        self.tardiness = 0
        self.time_exceeded = 0
        self.is_done = False
        
    def __lt__(self, other):
        # Define the comparison first by estimated_tardiness descending and then by index ascending
        if self.is_done and not other.is_done:
            return False
        if not self.is_done and other.is_done:
            return True
        
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
        self.schedule_heatmap = None
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

        self.cost_deadline = 0
        self.cost_hole = 0
        self.cost_processing = 0
        self.cost_makespan = 0

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
        self.schedule_heatmap = np.zeros(
            (len(self.machines), 150), dtype=np.int8)

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

        self.cost_deadline = 0
        self.cost_hole = 0
        self.cost_processing = 0
        self.cost_makespan = 0

        return self.get_observation(), self.get_info() 
    
    def action_masks(self):
        return self.action_mask

    def _update_action_masks(self, action):
        self._update_legal_actions(action)
        result = []  # 빈 리스트
        for legal_action in self.legal_actions:
            result.extend(legal_action)  # 각 legal_action을 result에 추가
        self.action_mask = np.array(result)  # 
        return self.action_mask

    def update_state(self, action=None):
        if action is not None:
            self.valid_count += 1
            self._schedule_operation(action)
            self._update_job_state()
            self._update_schedule_buffer()
            self._update_action_masks(action)
            self._update_machine_state(action)
            self.last_finish_time = self._get_final_operation_finish()
        else:
            self._update_schedule_buffer()
            self._update_machine_state(action)
            self._update_job_state()
            self._update_action_masks(action)


    def _update_legal_actions(self, action = None):
        # for job_index in range(len(self.jobs)):
        #     # 1. 선택된 Job의 모든 Operation가 이미 종료된 경우
        #     if self.schedule_buffer[job_index] == [-1, -1]:
        #         self.legal_actions[:, job_index] = False
        if action:
            for machine_index in range(len(self.machines)):
                # 2. 선택된 Machine가 선택된 Job의 Operation의 Type을 처리할 수 없는 경우
                machine = self.machines[machine_index]
                job_index = action[1]
                if self.schedule_buffer[job_index] == [-1, -1]:
                    self.legal_actions[:, job_index] = False
                    break
            
                operation_index = self.schedule_buffer[job_index][1]
                operation = self.job_infos[job_index].operation_queue[operation_index]
                if machine.can_process_operation(operation.type):
                    self.legal_actions[machine_index, job_index] = True
                else:
                    self.legal_actions[machine_index, job_index] = False
        else:
            for machine_index in range(len(self.machines)):
                # 2. 선택된 Machine가 선택된 Job의 Operation의 Type을 처리할 수 없는 경우
                machine = self.machines[machine_index]
                for job_index in range(len(self.jobs)):
                    if self.schedule_buffer[job_index] == [-1, -1]:
                        self.legal_actions[:, job_index] = False
                        continue
                    operation_index = self.schedule_buffer[job_index][1]
                    operation = self.job_infos[job_index].operation_queue[operation_index]
                    if machine.can_process_operation(operation.type):
                        self.legal_actions[machine_index, job_index] = True
                    else:
                        self.legal_actions[machine_index, job_index] = False

    def _update_machine_state(self, action=None):
        if action is None:
            for i, machine in enumerate(self.machines):
                self.machine_types[i] = [
                    1 if i in machine.ability else 0 for i in range(25)]
            return
        
        machine = self.machines[action[0]]
        operation_schedule = machine.operation_schedule
        self.schedule_heatmap[action[0]] = np.array(self._schedule_to_array(operation_schedule))

        # 선택된 리소스의 스케줄링된 Operation들
        if machine.operation_schedule:
            operation_time = sum([operation.duration for operation in machine.operation_schedule])
            machine.operation_rate = operation_time / self._get_final_operation_finish()
            self.machine_operation_rate[action[0]] = machine.operation_rate


        ####
        # for r, machine in enumerate(self.machines):
        #     operation_schedule = machine.operation_schedule
        #     self.schedule_heatmap[r] = self._schedule_to_array(
        #         operation_schedule)

        #     # 선택된 리소스의 스케줄링된 Operation들
        #     if machine.operation_schedule:
        #         operation_time = sum(
        #             [operation.duration for operation in machine.operation_schedule])

        #         machine.operation_rate = operation_time / self._get_final_operation_finish()
        #         self.machine_operation_rate[r] = machine.operation_rate

    def _update_job_state(self):
        for job_list in self.jobs:
            for job in job_list:
                remaining_operations = [op for op in job.operation_queue if op.finish is None]
                
                if not remaining_operations:
                    job.tardiness = job.operation_queue[-1].finish - job.deadline
                    job.time_exceeded = max(0, job.operation_queue[-1].finish - job.deadline)
                    job.estimated_tardiness = job.tardiness
                    job.is_done = True
                    continue
                
                earliest_operation = remaining_operations[0]
                best_finish_times = [
                    machine.cal_best_finish_time(op_earliest_start=earliest_operation.earliest_start, op_type = earliest_operation.type, op_duration = earliest_operation.duration)
                    for machine in self.machines
                ]
                best_finish_times = [time for time in best_finish_times if time != -1]

                if best_finish_times:
                    best_finish_time = min(best_finish_times)
                else:
                    best_finish_time = 0

                remaining_durations = [op.duration for op in remaining_operations[1:]]
                operation_deadline = job.deadline - sum(remaining_durations)
                
                job.estimated_tardiness = (best_finish_time - operation_deadline) #/ job.total_duration

        # Rebuild the heap based on the updated estimated tardiness values
        for job_list in self.jobs:
            heapq.heapify(job_list)


    def _schedule_to_array(self, operation_schedule, max_time = 150):
        def is_in_idle_time(time):
            for operation in operation_schedule:
                # 머신이 일하고 있는 시간에는 True 반환
                if operation.start <= time < operation.finish:
                    return True
            # 머신이 일하고 있는 시간에는 True 반환
            return False

        result = []

        for i in range(max_time):
            result.append(is_in_idle_time(i*100))

        return result

    def _update_schedule_buffer(self):
        # Clear the current schedule buffer

        for i in range(len(self.schedule_buffer)):
            if all(job.is_done for job in self.jobs[i]):
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

        # Update the earliest_start for the next operation in the job
        current_op_index = selected_job.operation_queue.index(selected_operation)
        if current_op_index + 1 < len(selected_job.operation_queue):
            next_operation = selected_job.operation_queue[current_op_index + 1]
            next_operation.earliest_start = selected_operation.finish
            return

    def get_observation(self):
        mean_estimated_tardiness_per_job = []
        std_estimated_tardiness_per_job = []
        for job_list in self.jobs:
            total_duration = job_list[0].total_duration
            estimated_tardiness = [job.estimated_tardiness / total_duration for job in job_list]
            mean_estimated_tardiness_per_job.append(np.mean(estimated_tardiness))
            std_estimated_tardiness_per_job.append(np.std(estimated_tardiness))

        observation = {
            'action_masks': self.action_mask,
            'job_details': self.current_job_details,
            #'machine_operation_rate': self.machine_operation_rate,
            #'machine_types': self.machine_types,
            'schedule_heatmap': self.schedule_heatmap,
            'schedule_buffer_job_repeat': np.array([elem[0] for elem in self.schedule_buffer]),
            'schedule_buffer_operation_index':  np.array([elem[1] for elem in self.schedule_buffer]),
            'mean_estimated_tardiness_per_job': np.array(mean_estimated_tardiness_per_job),
            'std_estimated_tardiness_per_job' : np.array(std_estimated_tardiness_per_job)
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
            'job_estimated_tardiness': [job.estimated_tardiness for job_list in self.jobs for job in job_list],
            'current_schedule': self.current_schedule,
            'job_deadline': [job.deadline for job_list in self.jobs for job in job_list],
            'job_time_exceeded': [job.time_exceeded for job_list in self.jobs for job in job_list],
            'job_tardiness': [job.tardiness for job_list in self.jobs for job in job_list],
            'cost_deadline': self.cost_deadline,
            'cost_hole': self.cost_hole,
            'cost_processing': self.cost_processing,
            'cost_makespan': self.cost_makespan
        }
    
    # def render_simple(self, num_steps=0, save_path="simple_schedule_plot.png"):
    #     current_schedule = [operation.to_dict() for operation in self.current_schedule]
    #     scheduled_df = list(filter(lambda operation: operation['sequence'] is not None, current_schedule))
    #     scheduled_df = pd.DataFrame(scheduled_df)

    #     if scheduled_df.empty:
    #         fig = Figure(figsize=(10, 5))
    #         canvas = FigureCanvas(fig)
    #         ax = fig.add_subplot(111)
    #         ax.set_title("Simple Operation Schedule Visualization")
    #         canvas.print_figure(save_path)
    #         return

    #     n_machines = len(self.machines)

    #     fig = Figure(figsize=(10, 5), dpi=80)  # 해상도 낮춤
    #     canvas = FigureCanvas(fig)
    #     ax = fig.add_subplot(111)
    #     ax.set_title(f'Simple Operation Schedule Visualization | steps = {num_steps}')
    #     ax.set_yticks(range(n_machines))
    #     ax.set_yticklabels([f'Machine {i}' for i in range(n_machines)])
        
    #     ax.set_xlim(0, self.last_finish_time // 100)
    #     ax.set_ylim(-1, n_machines)

    #     patches = []
    #     for i in range(len(self.machines)):
    #         machine_operations = scheduled_df[scheduled_df['machine'] == i]
    #         for index, operation in machine_operations.iterrows():
    #             start = operation["start"] // 100
    #             finish = operation["finish"] // 100
    #             color = mcolors.to_rgba(operation['color'])
    #             block = mpatches.Rectangle(
    #                 (start, i - 0.5), finish - start, 1, facecolor=color, edgecolor='black', linewidth=1)
    #             patches.append(block)

    #     for patch in patches:
    #         ax.add_patch(patch)

    #     fig.tight_layout()
    #     canvas.print_figure(save_path)
    #     plt.close(fig)
    
    # def render_image_to_array(self, num_steps = 0):
    #     buffer = io.BytesIO()
    #     self.render_simple(num_steps = num_steps, save_path=buffer)  # render_simple에서 save_path를 buffer로 설정
    #     buffer.seek(0)
    #     image = Image.open(buffer)
    #     image = image.resize((128, 128))  # 이미지 크기 조정
    #     image_array = np.array(image)
    #     buffer.close()
    #     return image_array
    
    def render(self, mode="seaborn", num_steps=0):
        current_schedule = [operation.to_dict() for operation in self.current_schedule]
        scheduled_df = list(filter(lambda operation: operation['sequence'] is not None, current_schedule))
        scheduled_df = pd.DataFrame(scheduled_df)

        if scheduled_df.empty:
            plt.figure(figsize=(12, 6))
            plt.title("Operation Schedule Visualization")
            return plt

        n_machines = len(self.machines)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f'Operation Schedule Visualization | steps = {num_steps}')
        ax.set_yticks(range(n_machines))
        ax.set_yticklabels([f'Machine {i}\n ability:{self.machines[i].ability}' for i in range(n_machines)])

        ax.set_xlim(0, self.last_finish_time)
        ax.set_ylim(-1, n_machines)

        legend_jobs = []

        for i in range(len(self.machines)):
            machine_operations = scheduled_df[scheduled_df['machine'] == i]
            for index, operation in machine_operations.iterrows():
                base_color = mcolors.to_rgba(operation['color'])
                job_index = operation["job_index"]
                shade_factor = (job_index + 1) / (len(self.jobs[0]) + 1)
                shaded_color = tuple([min(max(shade * shade_factor, 0), 1) if idx < 3 else shade for idx, shade in enumerate(base_color)])

                # Build operation sequence string
                operation_sequences = scheduled_df[(scheduled_df['job'] == operation['job']) & (scheduled_df['job_index'] == operation['job_index'])].sort_values(by='sequence')
                operation_info = ' -> '.join(map(str, operation_sequences['machine'].tolist()))

                job_label = f'Job {int(operation["job"]) + 1} - Repeat{job_index + 1} - ({operation_info})'
                legend = (int(operation["job"]) + 1, job_label, shaded_color)
                if legend not in legend_jobs:
                    legend_jobs.append(legend)

                op_block = mpatches.Rectangle(
                    (operation["start"], i - 0.5), operation["finish"] - operation["start"], 1, facecolor=shaded_color, edgecolor='black', linewidth=1)
                ax.add_patch(op_block)

        # Add legend for job repetition
        legend_jobs.sort()
        legend_patches = []
        for _, label, color in legend_jobs:
            legend_patches.append(mpatches.Patch(color=color, label=label))
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')

        plt.show()

    def is_legal(self, action):
        return self.action_mask[action[0]*len(self.jobs) + action[1]]
        return self.legal_actions[action[0], action[1]]

    def is_done(self):
        # 모든 Job의 Operation가 종료된 경우 Terminated를 True로 설정한다
        # 또한 legal_actions가 전부 False인 경우도 Terminated를 True로 설정한다
        return not np.any(self.action_mask) or all([job.operation_queue[-1].finish is not None for job_list in self.jobs for job in job_list])
    
    def _get_final_operation_finish(self):
        return max(self.current_schedule, key=lambda x: x.finish).finish

    def calculate_step_reward(self):
        # self.machine_term = 0.0
        # if np.any(self.machine_operation_rate):
        #     self.machine_term = np.mean(self.machine_operation_rate)
        # return self.machine_term        
        return 0.0
        # Schedule Buffer에 올라온 Job 들의 Estimated Tardiness 평균에 -1을 곱한 것을 반환
        return -np.mean([job_list[0].estimated_tardiness for job_list in self.jobs])

    def calculate_final_reward(self, total_durations, cost_deadline_per_time, cost_hole_per_time, cost_processing_per_time, cost_makespan_per_time, target_time = 1000):
        profit = (total_durations / 100) * 10
        cost = self.cal_final_cost(cost_deadline_per_time, cost_hole_per_time, cost_processing_per_time, cost_makespan_per_time)
        return ((profit - cost) / profit) * 100
        
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
            return np.mean([machine.operation_rate for machine in self.machines])
            # 각 Machine의 Hole을 점수로 만든다
            # return sum([machine.operation_rate for machine in self.machines]) / len(self.machines)
        
        def job_deadline_to_reward():
            sum_of_late_rate = 0
            total_job_length = 0
            for job_list in self.jobs:
                total_job_length += len(job_list)
                for job in job_list:
                    sum_of_late_rate += (job.tardiness / job.deadline)
            sum_of_late_rate /= total_job_length
            
            return 1 - sum_of_late_rate
            
        #def mixed_reward(target_time):
            # exceed_list = []
            # early_list = []
            # total_job_length = 0
            # for job_list in self.jobs:
            #     total_job_length += len(job_list)
            #     for job in job_list:
            #         exceed_list.append(job.time_exceeded / job.deadline)
            #         early_list.append(-1 * (job.tardiness / job.deadline))

            # m = np.mean(early_list)
            # s = np.std(early_list)

            # sum_of_late_rate = 0
            # for e in exceed_list:
            #     if e > 0:
            #         sum_of_late_rate += max(e, (e - m) / s)

            # sum_of_late_rate /= total_job_length
            
            # return (1 - sum_of_late_rate) * max(0.5, 1 - ((self._get_final_operation_finish() - target_time) / target_time))

            # sum_of_early_rate = 0
            # total_job_length = 0
            # for job_list in self.jobs:
            #     total_job_length += len(job_list)
            #     for job in job_list:
            #         sum_of_early_rate -= (job.tardiness / job.deadline)

            # sum_of_early_rate /= total_job_length
            # return max(0.25, sum_of_early_rate) * max(0.25, 1 - ((self._get_final_operation_finish() - target_time) / target_time))
            

        final_reward_by_op_rate = weight_op_rate * operation_rate_to_reward()
        final_reward_by_final_time = weight_final_time * final_time_to_reward(target_time) 
        final_reward_by_job_deadline = weight_job_deadline * job_deadline_to_reward()
        #final_reward_by_mixed = 100 * mixed_reward(target_time)

        return final_reward_by_op_rate + final_reward_by_final_time + final_reward_by_job_deadline #+ final_reward_by_mixed


    def cal_final_cost(self, cost_deadline_per_time = 5, cost_hole_per_time = 1, cost_processing_per_time = 2, cost_makespan_per_time = 10):
        # 반도체 공장
        # 주문: 만들어야하는 chip의 종류가 1~12이고 하루에 처리해야하는 일의 양이 변동함. (정규분포 따름)
        # 기계: 1~8번까지 있고, 각 기계가 할 수 있는 일의 양이 다름
        # Cost 
        # 

        # 1. Job deadline 어긴 정도 / 비율로 duration이 분모로감
        # - 총 duration 600, deadline : 900
        # - 끝난 시간 1000이면 1/6 * 단위시간 당 cost 만큼 cost 발생
        # -> obs 추가
        def cal_job_deadline_cost():
            job_deadline_cost = 0

            #sum_of_late_rate = 0
            sum_of_tard = 0
            #total_job_length = 0
            for job_list in self.jobs:
                #total_job_length += len(job_list)
                for job in job_list:
                    #total_duration = sum([op.duration for op in job.operation_queue])
                    sum_of_tard += job.time_exceeded
                    #sum_of_late_rate += (job.time_exceeded / total_duration)
            job_deadline_cost = sum_of_tard / 100 * cost_deadline_per_time #sum_of_late_rate * cost_deadline_per_time 

            self.cost_deadline = job_deadline_cost

            return job_deadline_cost

        # 2. machine Processing cost, hole cost 는 절반(hyperparams)으로
        # - 100 times 일할때마다 0.1,  
        # - 100 times 놀 때마다 0.05, 
        # 시작되는 시점부터 머신은 가동됨
        def cal_machine_cost():
            sum_of_hole_time = 0
            sum_of_up_time = 0
            for machine in self.machines:
                up_time = 0
                hole_time = 0
                if not machine.operation_schedule:
                    continue
                machine.operation_schedule.sort(key = lambda x: x.start)
                first_start = machine.operation_schedule[0].start
                last_finish = machine.operation_schedule[-1].finish
                # print(first_start)
                # print(last_finish)
            

                for op in machine.operation_schedule:
                    up_time += op.duration

                # print(up_time)
                hole_time = last_finish - first_start - up_time
                # print(hole_time)
                sum_of_up_time += up_time
                sum_of_hole_time += hole_time
            
            self.cost_hole = sum_of_hole_time * cost_hole_per_time / 100
            self.cost_processing = sum_of_up_time * cost_processing_per_time / 100

            return (sum_of_hole_time * cost_hole_per_time + sum_of_up_time * cost_processing_per_time) / 100
        
        # 3. Entire cost, (makespan cost)
        # - Makespan 100단위당 0.01
        def cal_entire_cost():
            self.cost_makespan = self._get_final_operation_finish() * cost_makespan_per_time / 100
            return self.cost_makespan
        
        # 각 cost를 알아보기 좋게 출력
        # print(f"Job Deadline Cost: {cal_job_deadline_cost()}")
        # print(f"Machine Cost: {cal_machine_cost()}")
        # print(f"Entire Cost: {cal_entire_cost()}")

        return cal_job_deadline_cost() + cal_machine_cost() + cal_entire_cost()