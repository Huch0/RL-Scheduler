import gymnasium as gym
from gymnasium import spaces

import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


class Resource():
    def __init__(self):
        self.task_schedule = [] # (order name, start_timing, finish_timing) -> (tasks)
        self.name = ""
        self.abilty = [] #task 1 ~ all
        self.reward = 0

class Order():
    def __init__(self):
        self.name = ""
        self.color = ""
        self.earliest_start = 0
        self.task_queue = [] # (tasks)
        self.reward = 0

class Task():
    def __init__(self):
        self.order = ""
        self.task_type = ""
        self.step = 0
        self.duration = 0
        self.predecessor = 0
        self.start = 0
        self.finish = 0

class SchedulingEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render.modes": ["seaborn"]}
    #resources_json, orders_json,
    def __init__(self, tasks, render_mode="seaborn"):
        super(SchedulingEnv, self).__init__()

        # Find the maximum 'resource' and 'predecessor' values in the tasks list
        self.resource_list = []
        max_resource = max(tasks, key=lambda task: task['resource'])[
            'resource']
        max_predecessor = max(
            tasks, key=lambda task: task['predecessor'] if task['predecessor'] is not None else 0)['predecessor']

        self.original_tasks = tasks
        self.num_tasks = len(tasks)

        self.action_space = spaces.MultiDiscrete(len(self.resource_list))
        self.observation_space = spaces.Dict({
            'sequence': spaces.Box(low=-1, high=self.num_tasks, shape=(self.num_tasks,), dtype=np.int32),
            'resource': spaces.Box(low=0, high=max_resource, shape=(self.num_tasks,), dtype=np.int32),
            'predecessor': spaces.Box(low=-1, high=max_predecessor, shape=(self.num_tasks,), dtype=np.int32),
            'earliest_start': spaces.Box(low=-1, high=5000, shape=(self.num_tasks,), dtype=np.int32),
            'duration': spaces.Box(low=0, high=5000, shape=(self.num_tasks,), dtype=np.int32),
            'start': spaces.Box(low=-1, high=5000, shape=(self.num_tasks,), dtype=np.int32),
            'finish': spaces.Box(low=-1, high=5000, shape=(self.num_tasks,), dtype=np.int32),
        })
        self.current_schedule = copy.deepcopy(tasks)
        self.num_scheduled_tasks = 0
        self.num_steps = 0

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)

        self.current_schedule = copy.deepcopy(self.original_tasks)
        self.num_scheduled_tasks = 0
        self.num_steps = 0

        return self._get_observation(), {}  # empty info dict

    def step(self, action):
        if action < 0 or action >= self.num_tasks:
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )
        ready_tasks = self._possible_schedule_list()
        self.num_steps += 1

        invalid_action = False

        if self.current_schedule[action]['sequence'] is not None or action not in ready_tasks:
            invalid_action = True

        if not invalid_action:
            self._schedule_task(action)
            
            reward = 0
        else:
            reward = -1

        terminated = bool(self.num_scheduled_tasks == self.num_tasks)

        if terminated:
            reward = self._calculate_reward()

        truncated = bool(self.num_steps == 1000)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

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
        # Create a DataFrame to store task scheduling information
        scheduled_df = list(
            filter(lambda task: task['sequence'] is not None, self.current_schedule))
        scheduled_df = pd.DataFrame(scheduled_df)

        if scheduled_df.empty:
            # Create an empty chart
            plt.figure(figsize=(12, 6))
            plt.title("Task Schedule Visualization")
            return plt

        # Create a bar plot using matplotlib directly
        fig, ax = plt.subplots(figsize=(12, 6))
        for resource in self.resources:
            resource_tasks = scheduled_df[scheduled_df['resource'] == resource]

            # Discriminate rows by lines
            line_offset = resource - 0.9  # Adjust the line offset for better visibility

            for index, task in resource_tasks.iterrows():
                ax.bar(
                    # Adjust 'x' to start from 'start'
                    x=task["start"] + task["duration"] / 2,
                    height=0.8,  # Height of the bar
                    width=task["duration"],  # Width of the bar
                    bottom=line_offset,  # Discriminate rows by lines
                    color=task['color'],
                    alpha=0.7,  # Transparency
                    label=f'Task {int(task["index"])}',  # Label for the legend
                )

        # Set y-axis ticks to show every resource
        ax.set_yticks(np.arange(0, len(self.resources)))
        ax.set_yticklabels(self.resources)

        ax.set(ylabel="Resource", xlabel="Time")
        # Place the legend outside the plot area
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title("Task Schedule Visualization")
        # 경고 무시 설정
        plt.rcParams['figure.max_open_warning'] = 0

    
        return fig

    def close(self):
        pass

    def _possible_schedule_list(self):
        latest_task_time = 0
        ready_tasks = []
        for i in range(len(self.current_schedule)):
            if self.current_schedule[i]['earliest_start'] is None:
                # Look at predecessor finish date
                predecessor = self.current_schedule[i]['predecessor']

                for pred_task in self.current_schedule:
                    if pred_task['index'] == predecessor:
                        # Predecessor is found
                        # See if it is finished.
                        # If so, task is good to go
                        if pred_task['finish'] is None:
                            break
                        else:
                            # it is finished
                            self.current_schedule[i]['earliest_start'] = pred_task['finish']
                            ready_tasks.append(i)

            elif self.current_schedule[i]['finish'] is None:
                ready_tasks.append(i)

        return ready_tasks

    def _schedule_task(self, action):
        # Implement the scheduling logic based on the action
        # You need to update the start and finish times of the tasks
        # based on the selected task index (action) and the current state.

        # Example: updating start and finish times
        selected_task = self.current_schedule[action]
        task_earliest_start = selected_task['earliest_start']
        task_index = selected_task['index']
        task_duration = selected_task['duration']
        task_resource = selected_task['resource']

        last_sequence = 0
        all_scheduled = list(
            filter(lambda task: task['sequence'] is not None, self.current_schedule))

        if all_scheduled:
            last_sequence = max(all_scheduled, key=lambda task: task['sequence'])[
                'sequence']

        resource_tasks = sorted(list(filter(lambda task: task['resource'] == task_resource
                                            and task['finish'] is not None, self.current_schedule)), key=lambda task: task['start'])

        open_windows = []
        start_window = 0
        last_alloc = 0

        for scheduled_task in resource_tasks:
            resource_init = scheduled_task['start']

            if resource_init > start_window:
                open_windows.append([start_window, resource_init])
            start_window = scheduled_task['finish']

            last_alloc = max(last_alloc, start_window)

        # Fit the task within the first possible window
        window_found = False
        if task_earliest_start is None:
            task_earliest_start = 0

        for window in open_windows:
            # Task could start before the open window closes
            if task_earliest_start <= window[1]:
                # Now let's see if it fits there
                potential_start = max(task_earliest_start, window[0])
                if potential_start + task_duration <= window[1]:
                    # Task fits into the window
                    min_earliest_start = potential_start
                    window_found = True
                    break

        # If no window was found, schedule it after the end of the last task on the resource
        if not window_found:
            min_earliest_start = max(task_earliest_start or 0, last_alloc)

        # Search the schedule plan, find the task and schedule it
        for i in range(0, len(self.current_schedule)):
            if self.current_schedule[i]['index'] == task_index:
                self.current_schedule[i]['sequence'] = last_sequence + 1
                self.current_schedule[i]['start'] = min_earliest_start
                self.current_schedule[i]['finish'] = min_earliest_start + \
                    task_duration
                break

        self.num_scheduled_tasks += 1
        return

    def _calculate_reward(self):
        # Implement your reward function based on the current state.
        # You can use the start and finish times of tasks to calculate rewards.
        # Example: reward based on minimizing the makespan
        makespan = max(self.current_schedule,
                       key=lambda x: x['finish'])['finish']
        return -makespan  # Negative makespan to convert it into a minimization problem

    def _get_observation(self):
        ready_tasks = self._possible_schedule_list()

        observation = {
            'sequence': np.array([task['sequence'] if task['sequence'] is not None else -1 for task in self.current_schedule], dtype=np.int32),
            'resource': np.array([task['resource'] for task in self.current_schedule], dtype=np.int32),
            'predecessor': np.array([task['predecessor'] if task['predecessor'] is not None else -1 for task in self.current_schedule], dtype=np.int32),
            'earliest_start': np.array([task['earliest_start'] if task['earliest_start'] is not None else -1 for task in self.current_schedule], dtype=np.int32),
            'duration': np.array([task['duration'] for task in self.current_schedule], dtype=np.int32),
            'start': np.array([task['start'] if task['start'] is not None else -1 for task in self.current_schedule], dtype=np.int32),
            'finish': np.array([task['finish'] if task['finish'] is not None else -1 for task in self.current_schedule], dtype=np.int32),
        }
        return observation


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    from helpers import load_orders

    env = SchedulingEnv(tasks=load_orders("../orders/orders-default.json"))
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)

    obs, _ = env.reset()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    step = 0
    while True:
        step += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        #env.render()
        # print(action, reward, step)
        # print("obs=", obs, "reward=", reward, "done=", done)
        if done:
            print("Goal reached!", "reward=", reward)
            env.render()
            break
