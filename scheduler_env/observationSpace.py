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


class ObservationSpaceInfo():
    def __init__(self, len_machine, len_jobs, max_time):
        self.len_machines = len_machine
        self.len_jobs = len_jobs
        self.max_time = max_time
        self.observation_space = spaces.Dict({
            # Vaild 행동, Invalid 행동 관련 지표
            "action_masks": spaces.Box(low=0, high=1, shape=(self.len_machines * self.len_jobs,), dtype=np.int8),
            # Instance 특징에 대한 지표
            "current_repeats": spaces.Box(low=0, high=20, shape=(self.len_jobs,), dtype=np.int64),
            'total_durations_per_job': spaces.Box(low=0, high=20, shape=(self.len_jobs,), dtype=np.int64),
            'num_operations_per_job': spaces.Box(low=0, high=20, shape=(self.len_jobs,), dtype=np.int64),
            'mean_operation_duration_per_job': spaces.Box(low=0, high=20, shape=(self.len_jobs,), dtype=np.float64),
            'std_operation_duration_per_job': spaces.Box(low=0, high=20, shape=(self.len_jobs,), dtype=np.float64),
            # 현 scheduling 상황 관련 지표
            'last_finish_time_per_machine': spaces.Box(low=0, high=max_time, shape=(self.len_machines,), dtype=np.int64),
            "machine_ability": spaces.Box(low=-1, high=100, shape=(self.len_machines,), dtype=np.int64),
            "hole_length_per_machine": spaces.Box(low=0, high=max_time, shape=(self.len_machines,), dtype=np.int64),
            "schedule_heatmap": spaces.Box(low=-1, high=2, shape=(self.len_machines, max_time), dtype=np.int8),
            "mean_real_tardiness_per_job": spaces.Box(low=-100, high=100, shape=(self.len_jobs,), dtype=np.float64),
            "std_real_tardiness_per_job": spaces.Box(low=-100, high=100, shape=(self.len_jobs,), dtype=np.float64),
            'remaining_repeats': spaces.Box(low=0, high=20, shape=(self.len_jobs,), dtype=np.int64),
            # schedule_buffer 관련 지표
            "schedule_buffer_job_repeat": spaces.Box(low=-1, high=10, shape=(self.len_jobs,), dtype=np.int64),
            "schedule_buffer_operation_index": spaces.Box(low=-1, high=10, shape=(self.len_jobs,), dtype=np.int64),
            "earliest_start_per_operation": spaces.Box(low=-1, high=max_time, shape=(self.len_jobs,), dtype=np.int64),
            'job_deadline': spaces.Box(low=-1, high=max_time, shape=(self.len_jobs,), dtype=np.int64),
            'op_duration': spaces.Box(low=-1, high=20, shape=(self.len_jobs,), dtype=np.int64),
            'op_type': spaces.Box(low=-1, high=25, shape=(self.len_jobs,), dtype=np.int64),
            # 추정 tardiness 관련 지표
            "mean_estimated_tardiness_per_job": spaces.Box(low=-100, high=100, shape=(self.len_jobs,), dtype=np.float64),
            "std_estimated_tardiness_per_job": spaces.Box(low=-100, high=100, shape=(self.len_jobs,), dtype=np.float64),
            # cost 관련 지표
            "cost_factor_per_time": spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float64),
            "current_costs": spaces.Box(low=0, high=50000, shape=(4,), dtype=np.float64),
        })
    
    def add_feature(self, feature_name, low, high, shape, dtype):
        """특정 feature를 추가합니다."""
        self.observation_space.spaces[feature_name] = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

    def remove_feature(self, feature_name):
        """특정 feature를 제거합니다."""
        if feature_name in self.observation_space.spaces:
            del self.observation_space.spaces[feature_name]
        else:
            print(f"Feature '{feature_name}' does not exist.")

    def modify_feature(self, feature_name, low=None, high=None, shape=None, dtype=None):
        """특정 feature의 속성을 변경합니다."""
        if feature_name in self.observation_space.spaces:
            feature = self.observation_space.spaces[feature_name]
            feature_low = low if low is not None else feature.low
            feature_high = high if high is not None else feature.high
            feature_shape = shape if shape is not None else feature.shape
            feature_dtype = dtype if dtype is not None else feature.dtype

            self.observation_space.spaces[feature_name] = spaces.Box(
                low=feature_low, high=feature_high, shape=feature_shape, dtype=feature_dtype
            )
        else:
            print(f"Feature '{feature_name}' does not exist.")

    def get_feature(self, feature_name):
        """특정 feature의 정보를 가져옵니다."""
        return self.observation_space.spaces.get(feature_name, f"Feature '{feature_name}' does not exist.")

    def get_observation_space(self):
        """전체 observation space를 반환합니다."""
        return self.observation_space
    
    
class ObservationSpace(ObservationSpaceInfo):
    def __init__(self, len_machine, len_jobs, max_time):
        super().__init__(len_machine, len_jobs, max_time)
        # 실제 데이터를 저장하는 딕셔너리
        self.observation_data = defaultdict(dict)

    def add_feature(self, feature_name, low, high, shape, dtype):
        """특정 feature를 추가합니다."""
        self.observation_space.spaces[feature_name] = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
        self.observation_data[feature_name] = np.zeros(shape, dtype=dtype)  # 기본 데이터를 추가합니다.

    def remove_feature(self, feature_name):
        """특정 feature를 제거합니다."""
        if feature_name in self.observation_space.spaces:
            del self.observation_space.spaces[feature_name]
            if feature_name in self.observation_data:
                del self.observation_data[feature_name]
        else:
            print(f"Feature '{feature_name}' does not exist.")

    def modify_feature(self, feature_name, low=None, high=None, shape=None, dtype=None):
        """특정 feature의 속성을 변경합니다."""
        if feature_name in self.observation_space.spaces:
            feature = self.observation_space.spaces[feature_name]
            feature_low = low if low is not None else feature.low
            feature_high = high if high is not None else feature.high
            feature_shape = shape if shape is not None else feature.shape
            feature_dtype = dtype if dtype is not None else feature.dtype

            self.observation_space.spaces[feature_name] = spaces.Box(
                low=feature_low, high=feature_high, shape=feature_shape, dtype=feature_dtype
            )
            self.observation_data[feature_name] = np.zeros(feature_shape, dtype=feature_dtype)  # 데이터를 초기화하거나 재할당합니다.
        else:
            print(f"Feature '{feature_name}' does not exist.")

    def get_feature(self, feature_name):
        """특정 feature의 정보를 가져옵니다."""
        return self.observation_space.spaces.get(feature_name, f"Feature '{feature_name}' does not exist.")

    def get_feature_data(self, feature_name):
        """특정 feature에 대한 실제 데이터를 가져옵니다."""
        return self.observation_data.get(feature_name, f"Feature data for '{feature_name}' does not exist.")

    def update_feature_data(self, feature_name, data):
        """특정 feature의 데이터를 업데이트합니다."""
        if feature_name in self.observation_data:
            self.observation_data[feature_name] = data
        else:
            print(f"Feature data for '{feature_name}' does not exist.")

    def get_observation_space(self):
        """전체 observation space를 반환합니다."""
        return self.observation_space

    def get_observation_data(self):
        """전체 observation data를 반환합니다."""
        return self.observation_data