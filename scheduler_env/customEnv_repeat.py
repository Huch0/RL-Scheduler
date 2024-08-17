import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
from stable_baselines3.common.env_checker import check_env
from scheduler_env.customScheduler_repeat import customRepeatableScheduler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # 필요한 모듈을 가져옵니다.
from collections import defaultdict

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space

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

    def _load_jobs_repeat(self, file):
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

    def __init__(self, machine_config_path, job_config_path, job_repeats_params, render_mode="seaborn", cost_deadline_per_time = 5, cost_hole_per_time = 1, cost_processing_per_time = 2, cost_makespan_per_time = 10, profit_per_time = 10, target_time = None, test_mode=False, max_time = 150, num_of_types = 4):
        super(SchedulingEnv, self).__init__()

        # cost 관련 변수
        self.cost_deadline_per_time = cost_deadline_per_time
        self.cost_hole_per_time = cost_hole_per_time
        self.cost_processing_per_time = cost_processing_per_time
        self.cost_makespan_per_time = cost_makespan_per_time

        # profit 관련 변수
        self.profit_per_time = profit_per_time

        self.target_time = target_time
        self.total_durations = 0
        
        self.job_repeats_params = job_repeats_params  # 각 Job의 반복 횟수에 대한 평균과 표준편차
        self.current_repeats = [job_repeat[0] for job_repeat in job_repeats_params]
        self.current_repeats_std = [job_repeat[1] for job_repeat in job_repeats_params]
        self.test_mode = test_mode
        self.best_makespan = float('inf')  # 최적 makespan

        self.jobs = self._load_jobs_repeat(job_config_path)
        self.machine_config = self._load_machines(machine_config_path)

        self.custom_scheduler = None

        self.len_machines = len(self.machine_config)
        self.len_jobs = len(self.jobs)

        self.num_steps = 0

        self.max_time = max_time

        self.mean_duration_per_job = None
        self.std_duration_per_job = None
        self.mean_deadline_per_job = None
        self.std_deadline_per_job = None
        self.num_operations_per_job = None

        self.mean_operation_duration_per_type = None
        self.std_operation_duration_per_type = None
        self.mappable_machine_count_per_type = None
        self.total_count_per_type = None
        self.num_of_types = num_of_types

        self.action_space = spaces.Discrete(self.len_machines * self.len_jobs)
        self.observation_space = spaces.Dict({
            # Vaild 행동, Invalid 행동 관련 지표
            "action_masks": spaces.Box(low=0, high=1, shape=(self.len_machines * self.len_jobs, ), dtype=np.int8),
            # Instance 특징에 대한 지표            
            # Operation Type별 지표
            "total_count_per_type": spaces.Box(low=-1, high=100, shape=(num_of_types, ), dtype=np.int64),
            "mappable_machine_count_per_type": spaces.Box(low=0, high=self.len_machines, shape=(num_of_types, ), dtype=np.int64),
            "mean_operation_duration_per_type": spaces.Box(low=0, high=20, shape=(num_of_types, ), dtype=np.float64),
            "std_operation_duration_per_type": spaces.Box(low=0, high=20, shape=(num_of_types, ), dtype=np.float64),
            # Job별 지표
            "current_repeats": spaces.Box(low=0, high=20, shape=(self.len_jobs, ), dtype=np.int64),
            'total_length_per_job' : spaces.Box(low=0, high=20, shape=(self.len_jobs, ), dtype=np.int64),
            'num_operations_per_job': spaces.Box(low=0, high=20, shape=(self.len_jobs, ), dtype=np.int64),
            'mean_operation_duration_per_job': spaces.Box(low=0, high=20, shape=(self.len_jobs, ), dtype=np.float64),
            'std_operation_duration_per_job': spaces.Box(low=0, high=20, shape=(self.len_jobs, ), dtype=np.float64),
            "mean_deadline_per_job": spaces.Box(low=-1, high=max_time, shape=(self.len_jobs, ), dtype=np.float64),
            "std_deadline_per_job": spaces.Box(low=-1, high=max_time, shape=(self.len_jobs, ), dtype=np.float64),
            # 현 scheduling 상황 관련 지표
            'last_finish_time_per_machine': spaces.Box(low=0, high=max_time, shape=(self.len_machines, ), dtype=np.int64),
            "machine_ability": spaces.Box(low=-1, high=100, shape=(self.len_machines, ), dtype=np.int64),
            "hole_length_per_machine": spaces.Box(low=0, high=max_time, shape=(self.len_machines, ), dtype=np.int64),
            "schedule_heatmap": spaces.Box(low=0, high=255, shape=(1, self.len_machines, max_time), dtype=np.uint8),
            "mean_real_tardiness_per_job": spaces.Box(low=-100, high=100, shape=(self.len_jobs, ), dtype=np.float64),
            "std_real_tardiness_per_job": spaces.Box(low=-100, high=100, shape=(self.len_jobs, ), dtype=np.float64),
            'remaining_repeats': spaces.Box(low=0, high=20, shape=(self.len_jobs, ), dtype=np.int64),
            # schedule_buffer 관련 지표
            "schedule_buffer_job_repeat": spaces.Box(low=-1, high=10, shape=(self.len_jobs, ), dtype=np.int64),
            "schedule_buffer_operation_index": spaces.Box(low=-1, high=10, shape=(self.len_jobs, ), dtype=np.int64),
            "earliest_start_per_operation": spaces.Box(low=-1, high=max_time, shape=(self.len_jobs, ), dtype=np.int64),
            'job_deadline': spaces.Box(low=-1, high=max_time, shape=(self.len_jobs, ), dtype=np.int64),
            'op_duration': spaces.Box(low=-1, high=20, shape=(self.len_jobs, ), dtype=np.int64),
            'op_type': spaces.Box(low=-1, high=25, shape=(self.len_jobs, ), dtype=np.int64),
            # 추정 tardiness 관련 지표
            "mean_estimated_tardiness_per_job": spaces.Box(low=-100, high=100, shape=(self.len_jobs, ), dtype=np.float64),
            "std_estimated_tardiness_per_job": spaces.Box(low=-100, high=100, shape=(self.len_jobs, ), dtype=np.float64),
            'cur_estimated_tardiness_per_job': spaces.Box(low=-100, high=100, shape=(self.len_jobs, ), dtype=np.float64),
            # cost 관련 지표
            "cost_factor_per_time": spaces.Box(low=-100, high=100, shape=(4, ), dtype=np.float64),
            "current_costs": spaces.Box(low=0, high=50000, shape=(4, ), dtype=np.float64),
        })
    def is_image(self):
        print(is_image_space(self.observation_space["schedule_heatmap"]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._initialize_scheduler()
        self.num_steps = 0
        self.cal_env_info()
        self.cal_job_info()

        return self._get_observation(), self._get_info()
    
    # def test_cal_best_finish_time(self):
    #     self.custom_scheduler.test_cal_best_finish_time()

    # def test_cal_estimated_tardiness(self):
    #     self.custom_scheduler.test_cal_estimated_tardiness()

    def step(self, action):
        # Map the action to the corresponding machine and job
        selected_machine_id = action // self.len_jobs
        selected_job_id = action % self.len_jobs
        action = [selected_machine_id, selected_job_id]

        # error_action이 아니라면 step의 수를 증가시킨다
        self.num_steps += 1
        reward = 0.0

        if self._is_legal(action):
            #reward += self._calculate_step_reward(action)
            self._update_state(action)
        else:  # Illegal action
            reward = -0.5

        terminated = self._is_done()
        if terminated:
            final_makespan = self.custom_scheduler._get_final_operation_finish()
            self.best_makespan = min(self.best_makespan, final_makespan)  # Update the best makespan
            reward = self._calculate_final_reward()

        truncated = bool(self.num_steps == 10000)
        if truncated:
            reward = -100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _is_legal(self, action):
        return self.custom_scheduler.is_legal(action)

    def _is_done(self):
        return self.custom_scheduler.is_done()

    def _update_state(self, action):
        self.custom_scheduler.update_state(action)

    def update_repeat_stds(self, new_std):
        self.job_repeats_params = [(mean, new_std) for mean, _ in self.job_repeats_params]

    def _get_observation(self):
        observation = self.custom_scheduler.get_observation()
        #추가
        observation["total_count_per_type"] = np.array(self.total_count_per_type)
        observation["mappable_machine_count_per_type"] = np.array(self.mappable_machine_count_per_type)
        observation["mean_deadline_per_job"] = np.array(self.mean_deadline_per_job)
        observation["std_deadline_per_job"] = np.array(self.std_deadline_per_job)
        observation["mean_operation_duration_per_type"] = np.array(self.mean_operation_duration_per_type)
        observation["std_operation_duration_per_type"] = np.array(self.std_operation_duration_per_type)
        
        #기존 정보 계산 최적화
        observation["mean_operation_duration_per_job"] = np.array(self.mean_operation_duration_per_job)
        observation["std_operation_duration_per_job"] = np.array(self.std_operation_duration_per_job)
        observation["current_repeats"] = np.array(self.current_repeats)
        observation["num_operations_per_job"] = np.array(self.num_operations_per_job)
        observation["total_length_per_job"] = np.array([int(num * mean) for num, mean in zip(self.num_operations_per_job, self.mean_operation_duration_per_job)])

        return observation
    
    def set_test_mode(self, test_mode):
        self.test_mode = test_mode
        self.reset()

    # For MaskablePPO
    def action_masks(self):
        return self.custom_scheduler.action_masks()

    def _get_info(self):
        info = self.custom_scheduler.get_info()
        info['num_steps'] = self.num_steps
        info['current_repeats'] = self.current_repeats
        return info

    def _calculate_final_reward(self):
        return self.custom_scheduler.calculate_final_reward()
    
    def _calculate_step_reward(self, action):
        return self.custom_scheduler.calculate_step_reward(action)

    def _initialize_scheduler(self):
        if self.test_mode:
            repeats_list = self.current_repeats[::]
        # 각 Job의 반복 횟수를 랜덤하게 설정
        # 랜덤 반복 횟수에 따라 Job 인스턴스를 생성
        else:
            repeats_list = []
            for mean, std in self.job_repeats_params:
                repeats = max(1, int(np.random.normal(mean, std)))
                repeats_list.append(repeats)
            self.current_repeats = repeats_list[::]
            
        self._calculate_target_time()

        random_jobs = []
        for job, repeat in zip(self.jobs, repeats_list):
            random_job_info = {
                'name': job['name'],
                'color': job['color'],
                'operations': job['operations'],
                'deadline': job['deadline'][:repeat]  # 주어진 반복 횟수에 따라 deadline 설정
            }
            random_jobs.append(random_job_info)

        # 랜덤 Job 인스턴스를 사용하여 customScheduler 초기화
        self.custom_scheduler = customRepeatableScheduler(jobs=random_jobs, machines=self.machine_config, cost_deadline_per_time= self.cost_deadline_per_time, cost_hole_per_time = self.cost_hole_per_time, cost_processing_per_time = self.cost_processing_per_time, cost_makespan_per_time = self.cost_makespan_per_time, profit_per_time = self.profit_per_time, current_repeats=self.current_repeats, max_time=self.max_time)
        self.custom_scheduler.reset()

    def _calculate_target_time(self):
        total_duration = 0
        for i in range(len(self.jobs)):
            job_duration = sum(op['duration'] for op in self.jobs[i]['operations'])
            total_duration += job_duration * self.current_repeats[i]
        
        self.total_durations = total_duration
        self.target_time = total_duration / self.len_machines

    def render(self, mode="human"):
        self.custom_scheduler.render(mode=mode, num_steps=self.num_steps)

    def visualize_graph(self):
        self.custom_scheduler.visualize_graph()

    def print_result(self, info, detail_mode = False):
        current_repeats = info['current_repeats']
        print(f"Current Repeats\t\t\t:\t{current_repeats}")

        # 최종 점수 출력
        reward = info["reward"]
        print(f"Goal reached! Final score\t:\t{reward:.2f}")

        env = info["env"]

        cost_deadline = info["cost_deadline"]
        cost_hole = info["cost_hole"]
        cost_processing = info["cost_processing"]
        cost_makespan = info["cost_makespan"]
        sum_costs = cost_deadline + cost_hole + cost_processing + cost_makespan
        profit = env.total_durations / 100 * info["profit_ratio"]
        print(f"Total revenue\t\t\t:\t{profit:.2f} - {sum_costs:.2f} = {profit - sum_costs:.2f}")
        print(f"Sum of Costs\t\t\t:\t{cost_deadline + cost_hole + cost_processing + cost_makespan:.2f}")
        print(f"Cost Deadline\t\t\t:\t{cost_deadline:.2f}")
        print(f"Cost Hole\t\t\t:\t{cost_hole:.2f}")
        print(f"Cost Processing\t\t\t:\t{cost_processing:.2f}")
        print(f"Cost Makespan\t\t\t:\t{cost_makespan:.2f}")


        # 최종 완료 시간 출력    
        print(f"Finish Time / Target Time\t:\t{info['finish_time']} / {int(env.target_time)}")

        # jobs 생성
        jobs = info["jobs"]
        job_deadlines = info['job_deadline']
        job_tardiness = info['job_time_exceeded']
        index = 0

        for job_id in range(len(jobs)):
            jobs[job_id].sort(key = lambda x : x.index)

        if detail_mode:
            for job_list in jobs:
                for job in job_list:
                    print(str(job))

            # print("--------------------------------")
            # print(job_deadlines)
            # print("--------------------------------")
            # print(job_tardiness)
            # for repeat in current_repeats:
            #     for r in range(repeat):
            #         deadline = job_deadlines[index+r]
            #         tardiness = job_tardiness[index+r]
            #         print(f"Job {index + 1} - Repeat {r + 1}\t\t:\tTardiness/Deadline = {tardiness}/{deadline}")
            #     index += 1


        # Calculate Tardiness/Deadline ratios and assign colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                '#aec7e8', '#ffbb78']  # Color palette for jobs
        
        ratios = []
        tardinesses = [job.tardiness for job_list in jobs for job in job_list]
        x_labels = []
        bar_colors = []
        x_positions = []
        
        current_x = 0
        
        for job_id in range(len(current_repeats)):
            repeat = current_repeats[job_id]
            x_labels.extend([f'Job {job_id} - Repeat {i+1}' for i in range(repeat)])
            bar_colors.extend([colors[job_id - 1]] * repeat)
            x_positions.extend([current_x + i for i in range(repeat)])
            current_x += repeat + 1  # Add space between different jobs

        # Calculate and print the average Tardiness/Deadline ratio
        avg_tardiness = np.mean(tardinesses)
        print(f"Average Tardiness:\t{avg_tardiness:.2f}")
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_width = 0.8
        ax.bar(x_positions, tardinesses, width=bar_width, color=bar_colors)
        
        # Set x-ticks and x-tick labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=90, ha='center')
        ax.set_xlabel('Job - Repeat')
        ax.set_ylabel('Tardiness per Job repeat')
        ax.set_title('Tardiness per Job repeat')
        
        # Legend
        unique_jobs = list(set([f'{job.name}' for job_list in jobs for job in job_list]))
        legend_patches = [mpatches.Patch(color=colors[i], label=unique_jobs[i]) for i in range(len(unique_jobs))]
        legend_patches.sort(key=lambda x: int(x.get_label().split()[1]))
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

    def cal_env_info(self):
        self.mean_operation_duration_per_type = []
        self.std_operation_duration_per_type = []
        self.mappable_machine_count_per_type = []
        self.total_count_per_type = []

        # Operation Type별로 필요한 정보를 저장할 딕셔너리 생성
        operation_stats = defaultdict(lambda: {'count': 0, 'total_duration': [], 'machine_count': 0})

        # Job의 Operation을 순회하며 통계 정보 수집
        for job_info, repeat in zip(self.jobs, self.current_repeats):
            for operation in job_info["operations"]:
                op_type = operation["type"]
                operation_stats[op_type]['count'] += repeat  # 반복 횟수만큼 count 증가
                operation_stats[op_type]['total_duration'].extend([operation["duration"] // 100] * repeat)

        # 각 Operation Type을 처리할 수 있는 머신의 수 계산
        for machine in self.machine_config:
            for ability in machine["ability"]:
                operation_stats[ability]['machine_count'] += 1

        # 통계 정보 계산
        data = []
        for op_type, stats in sorted(operation_stats.items()):
            total_duration_array = np.array(stats['total_duration'])
            avg_duration = np.mean(total_duration_array)
            std_duration = np.std(total_duration_array)
            data.append({
                'Operation Type': op_type,
                'Total Count': stats['count'],
                'Avg Duration': avg_duration,
                'Std Duration': std_duration,
                'Machine Count': stats['machine_count']
            })

            self.mean_operation_duration_per_type.append(avg_duration) 
            self.std_operation_duration_per_type.append(std_duration)
            self.mappable_machine_count_per_type.append(stats['machine_count'])
            self.total_count_per_type.append(stats['count'])


        return data

    def show_env_info(self):
        data = self.cal_env_info()

        # DataFrame으로 변환하여 표 형식으로 출력
        df = pd.DataFrame(data)

        # 스타일링 적용
        styled_df = df.style.format({
            'Total Count': '{:,.0f}',
            'Avg Duration': '{:.2f}',
            'Std Duration': '{:.2f}',
            'Machine Count': '{:,.0f}'
        }).background_gradient(cmap='Blues')

        print(styled_df)
        return styled_df

    def cal_job_info(self):
        job_data = []

        self.mean_deadline_per_job = []
        self.std_deadline_per_job = []
        self.mean_operation_duration_per_job = []
        self.std_operation_duration_per_job = []
        self.num_operations_per_job = []

        for job_info, repeat in zip(self.jobs, self.current_repeats):
            durations = [op["duration"] // 100 for op in job_info["operations"]]
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            num_operations = len(job_info["operations"])
            deadlines = np.array(job_info["deadline"][:repeat]) // 100
            mean_deadline = np.mean(deadlines)
            std_deadline = np.std(deadlines)

            job_data.append({
                'Job Name': job_info["name"],
                'Mean Ops Duration': mean_duration,
                'Std Ops Duration': std_duration,
                '# of Ops': num_operations,
                'Mean Deadline': mean_deadline,
                'Std Deadline': std_deadline,
                'Repeats': repeat
            })
            
            self.mean_deadline_per_job.append(mean_deadline)
            self.std_deadline_per_job.append(std_deadline)
            self.mean_operation_duration_per_job.append(mean_duration)
            self.std_operation_duration_per_job.append(std_duration)
            self.num_operations_per_job.append(num_operations)

        return job_data

    def show_job_info(self):
        # Job별 통계 정보 수집
        job_data = self.cal_job_info()

        # DataFrame으로 변환하여 표 형식으로 출력
        job_df = pd.DataFrame(job_data)

        # 스타일링 적용
        styled_job_df = job_df.style.format({
            'Mean Duration': '{:.2f}',
            'Std Duration': '{:.2f}',
            '# of Ops': '{:,.0f}',
            'Mean Deadline': '{:.2f}',
            'Std Variance': '{:.2f}',
            'Repeats': '{:,.0f}'
        }).background_gradient(cmap='Greens')

        print(styled_job_df)
        return styled_job_df

if __name__ == "__main__":
    env_5_8_8_2 = SchedulingEnv(machine_config_path= "instances/Machines/v0-5.json", job_config_path = "instances/Jobs/v0-8x12-deadline.json", job_repeats_params = [(8, 2)] * 8)
    env = env_5_8_8_2
    check_env(env)

    step = 0
    obs, info = env.reset()
    print(info["current_repeats"])

    while True:
        step += 1
        action = env.action_space.sample()    
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        info["reward"] = reward
        info["env"] = env
        info["profit_ratio"] = env.profit_per_time
        
        if done:
            env.print_result(info, detail_mode = True)
            env.render()
            break