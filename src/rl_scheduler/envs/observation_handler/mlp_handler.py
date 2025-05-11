import numpy as np
from typing import Dict, Any, List
from gymnasium import spaces
from .observation_handler import ObservationHandler
from rl_scheduler.envs.action_handler import MJHandler


class MLPHandler(ObservationHandler):
    def __init__(self, scheduler, mj_action_handler: MJHandler, time_horizon: int):
        super().__init__(scheduler)

        self.M = len(self.scheduler.machine_templates)
        self.J = len(self.scheduler.job_templates)
        self.time_horizon = time_horizon
        self.mj_action_handler = mj_action_handler

        # operation types mapping
        self.type_set = {
            op_template.type_code
            for job_template in scheduler.job_templates
            for op_template in job_template.operation_template_sequence
        }
        # Get the total number of distinct operation types
        self.num_op_types = len(self.type_set)

        # Create a mapping from operation type code to index
        self.type_index = {
            type_code: idx for idx, type_code in enumerate(sorted(self.type_set))
        }

        # static observation features
        # 1-1 #
        self.op_type_counts = np.zeros(self.num_op_types, dtype=np.int64)
        # 1-2 #
        self.durations = [
            np.array([], dtype=np.float16) for _ in range(self.num_op_types)
        ]
        self.mean_op_duration = np.zeros(self.num_op_types, dtype=np.float16)
        self.std_op_duration = np.zeros(self.num_op_types, dtype=np.float16)
        # 2 #
        self.mean_job_deadline = np.zeros(self.J, dtype=np.float16)
        self.std_job_deadline = np.zeros(self.J, dtype=np.float16)
        # 3 #
        self.machine_ability = np.zeros(self.M, dtype=np.float16)
        # --- #

    def update_static_observation_features(self):
        # static observation features
        # These are calculated once when scheduler.reset() is called
        # and used for all steps

        # Initialize static observation features
        # 1-1 #
        self.op_type_counts = np.zeros(self.num_op_types, dtype=np.int64)
        # 1-2 #
        self.durations = [
            np.array([], dtype=np.float16) for _ in range(self.num_op_types)
        ]
        self.mean_op_duration = np.zeros(self.num_op_types, dtype=np.float16)
        self.std_op_duration = np.zeros(self.num_op_types, dtype=np.float16)
        # 2 #
        self.mean_job_deadline = np.zeros(self.J, dtype=np.float16)
        self.std_job_deadline = np.zeros(self.J, dtype=np.float16)
        # 3 #
        self.machine_ability = np.zeros(self.M, dtype=np.float16)
        # --- #

        scheduler = self.scheduler

        # 1-1 total count per operation type
        for job_template in scheduler.job_templates:
            for op_template in job_template.operation_template_sequence:
                self.op_type_counts[self.type_index[op_template.type_code]] += 1

        # 1-2 duration statistics per operation type
        for job_template in scheduler.job_templates:
            for op_template in job_template.operation_template_sequence:
                idx = self.type_index[op_template.type_code]
                # append duration to the corresponding type
                self.durations[idx] = np.append(
                    self.durations[idx], op_template.duration
                )

        # calculate mean and std for each type
        for idx in range(self.num_op_types):
            if len(self.durations[idx]) > 0:
                self.mean_op_duration[idx] = np.mean(self.durations[idx])
                self.std_op_duration[idx] = np.std(self.durations[idx])
            else:
                self.mean_op_duration[idx] = 0
                self.std_op_duration[idx] = 0

        # 2 deadline stats per job
        for jt_idx, job_template in enumerate(scheduler.job_instances):
            deadlines = [
                job_instance.profit_fn.deadline for job_instance in job_template
            ]
            self.mean_job_deadline[jt_idx] = float(np.mean(deadlines))
            self.std_job_deadline[jt_idx] = float(np.std(deadlines))

        # 3 machine ability
        def encode_ability(type_indices: List[int]) -> int:
            # Convert the list of supported type indices to a bitmask
            # For example, if supported_type_indices = [0, 2], the bitmask would be 101
            # which is 5 in decimal.
            code = 0
            for i in type_indices:
                code |= 1 << i
            return code

        for mt_idx, machine_template in enumerate(scheduler.machine_templates):
            supported_type_indices = [
                self.type_index[op_template.type_code]
                for op_template in machine_template.operation_template_sequence
            ]
            self.machine_ability[mt_idx] = encode_ability(supported_type_indices)

    def create_observation_space(self) -> spaces.Space:
        """
        Define the observation space for a dictionary-based observation.

        Note: This is a high-level placeholder implementation.
        The observation space should be redefined in detail later.
        """
        return spaces.Dict(
            {
                # Vaild 행동, Invalid 행동 관련 지표
                "action_masks": spaces.Box(
                    low=0, high=1, shape=(self.M * self.J,), dtype=np.int8
                ),
                # Instance 특징에 대한 지표
                # Operation Type별 지표
                "total_count_per_type": spaces.Box(
                    low=-1, high=50, shape=(self.num_op_types,), dtype=np.int16
                ),
                "mean_operation_duration_per_type": spaces.Box(
                    low=0, high=20, shape=(self.num_op_types,), dtype=np.float16
                ),
                "std_operation_duration_per_type": spaces.Box(
                    low=0, high=20, shape=(self.num_op_types,), dtype=np.float16
                ),
                # Job별 지표
                "mean_deadline_per_job": spaces.Box(
                    low=-1, high=self.time_horizon, shape=(self.J,), dtype=np.float16
                ),
                "std_deadline_per_job": spaces.Box(
                    low=-1, high=self.time_horizon, shape=(self.J,), dtype=np.float16
                ),
                # 현 scheduling 상황 관련 지표
                "last_finish_time_per_machine": spaces.Box(
                    low=0, high=self.time_horizon, shape=(self.M,), dtype=np.int16
                ),
                "machine_ability": spaces.Box(
                    low=-1, high=100, shape=(self.M,), dtype=np.int16
                ),
                "hole_length_per_machine": spaces.Box(
                    low=0, high=self.time_horizon, shape=(self.M,), dtype=np.int16
                ),
                "machine_utilization_rate": spaces.Box(
                    low=0, high=1, shape=(self.M,), dtype=np.float16
                ),
                "remaining_repeats": spaces.Box(
                    low=0, high=20, shape=(self.J,), dtype=np.int16
                ),
                # schedule_heatmap 관련 지표
                "schedule_heatmap": spaces.Box(
                    low=-1, high=2, shape=(self.M, self.time_horizon), dtype=np.int8
                ),
                # schedule_buffer 관련 지표
                "schedule_buffer_job_repeat": spaces.Box(
                    low=-1, high=10, shape=(self.J,), dtype=np.int16
                ),
                "schedule_buffer_operation_index": spaces.Box(
                    low=-1, high=10, shape=(self.J,), dtype=np.int16
                ),
                "cur_op_earliest_start": spaces.Box(
                    low=-1, high=self.time_horizon, shape=(self.J,), dtype=np.int16
                ),
                "cur_job_deadline": spaces.Box(
                    low=-1, high=self.time_horizon, shape=(self.J,), dtype=np.int16
                ),
                "cur_op_duration": spaces.Box(
                    low=-1, high=20, shape=(self.J,), dtype=np.int16
                ),
                "cur_op_type": spaces.Box(
                    low=-1, high=25, shape=(self.J,), dtype=np.int16
                ),
                "cur_remain_working_time": spaces.Box(
                    low=0, high=20, shape=(self.J,), dtype=np.int16
                ),
                "cur_remain_num_op": spaces.Box(
                    low=0, high=10, shape=(self.J,), dtype=np.int16
                ),
                # 추정 tardiness 관련 지표
                "mean_estimated_tardiness_per_job": spaces.Box(
                    low=-100, high=100, shape=(self.J,), dtype=np.float16
                ),
                "std_estimated_tardiness_per_job": spaces.Box(
                    low=-100, high=100, shape=(self.J,), dtype=np.float16
                ),
                "cur_estimated_tardiness_per_job": spaces.Box(
                    low=-100, high=100, shape=(self.J,), dtype=np.float16
                ),
                # cost 관련 지표
                "current_costs": spaces.Box(
                    low=0, high=50000, shape=(4,), dtype=np.float16
                ),
            }
        )

    def get_observation(self) -> Dict[str, Any]:
        """
        Generate a full observation dictionary matching create_observation_space.

        Returns
        -------
        dict
            A mapping from feature names to their current values:
            - "action_masks": (M*J,) int8
            - "total_count_per_type": (T,) int16
            - "mean_operation_duration_per_type": (T,) float16
            - "std_operation_duration_per_type": (T,) float16
            - "mean_deadline_per_job": (J,) float16
            - "std_deadline_per_job": (J,) float16
            - "last_finish_time_per_machine": (M,) int16
            - "machine_ability": (M,) int16
            - "hole_length_per_machine": (M,) int16
            - "machine_utilization_rate": (M,) float16
            - "remaining_repeats": (J,) int16
            - "schedule_heatmap": (M, time_horizon) int8
            - "schedule_buffer_job_repeat": (J,) int16
            - "schedule_buffer_operation_index": (J,) int16
            - "cur_op_earliest_start": (J,) int16
            - "cur_job_deadline": (J,) int16
            - "cur_op_duration": (J,) int16
            - "cur_op_type": (J,) int16
            - "cur_remain_working_time": (J,) int16
            - "cur_remain_num_op": (J,) int16
            - "mean_estimated_tardiness_per_job": (J,) float16
            - "std_estimated_tardiness_per_job": (J,) float16
            - "cur_estimated_tardiness_per_job": (J,) float16
            - "current_costs": (4,) float16
        """
        scheduler = self.scheduler

        # Dynamic observation features
        # These are calculated at each step

        # 1 action masks
        masks = self.mj_action_handler.compute_action_mask()

        # 2 Machine features
        # 2-1 last finish time per machine
        last_end_time = np.array(
            [machine.last_assigned_end_time for machine in scheduler.machine_instances],
            dtype=np.uint16,
        )
        # 2-2 machine hole length and utilization
        machine_uptime = np.array(
            [
                sum(op.duration for op in machine.assigned_operations)
                for machine in scheduler.machine_instances
            ],
            dtype=np.int16,
        )
        # Idle time = last finish time - uptime
        machine_idletime = last_end_time - machine_uptime
        # Utilization = uptime / last finish time (avoid div by zero)
        machine_util_rate = np.zeros_like(machine_idletime, dtype=np.float16)
        nonzero = last_end_time > 0
        machine_util_rate[nonzero] = machine_uptime[nonzero].astype(
            np.float16
        ) / last_end_time[nonzero].astype(np.float16)
        # 2-3 machine schedule heatmap
        schedule_heatmap = np.zeros((self.M, self.time_horizon), dtype=np.int8)
        for machine_id, machine in enumerate(scheduler.machine_instances):
            for op in machine.assigned_operations:
                start_time = op.start_time
                end_time = op.end_time
                schedule_heatmap[machine_id, start_time:end_time] = 1

        # 3 job buffer features
        # 3-1 remaining repeats
        remaining_repeats = np.array(
            [
                sum(not job_instance.completed for job_instance in job_type)
                for job_type in scheduler.job_instances
            ],
            dtype=np.int16,
        )
        priorities = self.mj_action_handler.priority_rule.assign_priority()
        # Job instances with highest priority
        # for each job type
        job_buffer = [
            job_type[priorities[jt_idx]]
            for jt_idx, job_type in enumerate(scheduler.job_instances)
        ]

        # 3-2 highest_priority_job_instance features
        # Get the highest priority job instance for each job type
        highest_priority_job_instance = np.array(priorities, dtype=np.int16)
        highest_priority_job_instance_deadline = np.array(
            [
                job_instance.profit_fn.deadline if not job_instance.completed else -1
                for job_instance in job_buffer
            ],
            dtype=np.int16,
        )
        highest_priority_job_instance_remaining_duration = np.array(
            [
                sum(
                    job_instance.operation_instance_sequence[i].duration
                    for i in range(
                        job_instance.next_op_idx,
                        len(job_instance.operation_instance_sequence),
                    )
                )
                for job_instance in job_buffer
            ],
            dtype=np.int16,
        )
        highest_priority_job_instance_num_remaining_ops = np.array(
            [
                len(job_instance.operation_instance_sequence) - job_instance.next_op_idx
                for job_instance in job_buffer
            ],
            dtype=np.int16,
        )

        # 3-3 next operation features
        # Get the first remaining operation index
        # for each highest priority job instance
        next_op_indices = np.array(
            [
                (
                    job_instance.next_op_idx if not job_instance.completed else -1
                )  # all job instances of this job type are completed
                for job_instance in job_buffer
            ],
            dtype=np.int16,
        )
        next_op_earliest_starts = np.array(
            [
                (
                    job_instance.operation_instance_sequence[
                        next_op_idx
                    ].earliest_start_time
                    if next_op_idx != -1
                    else -1
                )
                for job_instance, next_op_idx in zip(job_buffer, next_op_indices)
            ],
            dtype=np.int16,
        )
        next_op_durations = np.array(
            [
                (
                    job_instance.operation_instance_sequence[next_op_idx].duration
                    if next_op_idx != -1
                    else -1
                )
                for job_instance, next_op_idx in zip(job_buffer, next_op_indices)
            ],
            dtype=np.int16,
        )
        next_op_types = np.array(
            [
                (
                    self.type_index[
                        job_instance.operation_instance_sequence[next_op_idx].type_code
                    ]
                    if next_op_idx != -1
                    else -1
                )
                for job_instance, next_op_idx in zip(job_buffer, next_op_indices)
            ],
            dtype=np.int16,
        )

        return {
            "action_masks": masks,
            "total_count_per_type": self.op_type_counts,
            "mean_operation_duration_per_type": self.mean_op_duration,
            "std_operation_duration_per_type": self.std_op_duration,
            "mean_deadline_per_job": self.mean_job_deadline,
            "std_deadline_per_job": self.std_job_deadline,
            "last_finish_time_per_machine": last_end_time,
            "machine_ability": self.machine_ability,
            "hole_length_per_machine": machine_idletime,
            "machine_utilization_rate": machine_util_rate,
            "remaining_repeats": remaining_repeats,
            "schedule_heatmap": schedule_heatmap,
            "schedule_buffer_job_repeat": highest_priority_job_instance,
            "schedule_buffer_operation_index": next_op_indices,
            "cur_op_earliest_start": next_op_earliest_starts,
            "cur_job_deadline": highest_priority_job_instance_deadline,
            "cur_op_duration": next_op_durations,
            "cur_op_type": next_op_types,
            "cur_remain_working_time": highest_priority_job_instance_remaining_duration,
            "cur_remain_num_op": highest_priority_job_instance_num_remaining_ops,
            # "mean_estimated_tardiness_per_job": mean_est_tardy,
            # "std_estimated_tardiness_per_job": std_est_tardy,
            # "cur_estimated_tardiness_per_job": cur_est_tardy,
            # "current_costs": current_costs,
        }
