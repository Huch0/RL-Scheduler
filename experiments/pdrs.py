import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import torch as th
# from sb3_contrib import MaskablePPO
# from utils.tb_logger import CustomLogger
# from scheduler_env.pdr_env import SchedulingEnv as PDRenv
# from scheduler_env.benchmark_rl_env import SchedulingEnv as RLenv

### start : PDR 구현
def CR(env):
    """
    Critical Ratio
    - Pick the job with the smallest critical ratio
    (due date - current time) / total processing time remaining.
    """
    min_critical_ratio = np.inf
    selected_job = None
    jobs = env.custom_scheduler.jobs
    for job_list in jobs:
        for job in job_list:
            # Calculate the critical ratio
            due_date = job.deadline
            remaining_time = sum([op.duration for op in job.operation_queue if op.finish is None])
            critical_ratio = (due_date - 0) / remaining_time if remaining_time > 0 else np.inf

            if critical_ratio < min_critical_ratio:
                min_critical_ratio = critical_ratio
                selected_job = job

    selected_op = None
    for op in selected_job.operation_queue:
        if op.finish is None:
            selected_op = op
            break

    # Find the first available machine to schedule the job
    selected_machine = select_machine(env.custom_scheduler.machines, selected_job, selected_op)

    return {
        'selected_machine': selected_machine,
        'selected_job': selected_job,
        'selected_op': selected_op,
    }

def FDD_over_MWKR(env):
    """
    Flow Due Date / Most Work Remaining
    """
    min_fdd_mwkr = np.inf
    selected_job = None
    jobs = env.custom_scheduler.jobs
    for job_list in jobs:
        for job in job_list:
            scheduled_time = 0
            remaining_time = 0
            current_time = 0
            for op in job.operation_queue:
                if op.finish is not None:
                    scheduled_time += op.duration
                else:
                    if current_time == 0:
                        current_time = op.duration
                    remaining_time += op.duration

            if remaining_time == 0:
                continue

            # Calculate the FDD
            r = 0  # arrival time
            fdd = r + scheduled_time + current_time
            # Calculate the MWKR
            mwkr = remaining_time
            fdd_mwkr = fdd / mwkr

            if fdd_mwkr < min_fdd_mwkr:
                min_fdd_mwkr = fdd_mwkr
                selected_job = job

    selected_op = None
    for op in selected_job.operation_queue:
        if op.finish is None:
            selected_op = op
            break

    # Find the first available machine to schedule the job
    selected_machine = select_machine(env.custom_scheduler.machines, selected_job, selected_op)

    return {
        'selected_machine': selected_machine,
        'selected_job': selected_job,
        'selected_op': selected_op,
    }

def MWKR(env):
    """
    Most Work Remaining
    - Pick the job with the maximum total processing time remaining.

    :param env: The environment to schedule.

    :return action: The job to schedule next.
    """
    # Find the job with the most work remaining
    max_remaining_time = 0
    selected_job = None
    jobs = env.custom_scheduler.jobs
    for job_list in jobs:
        for job in job_list:
            remaining_time = sum([op.duration for op in job.operation_queue if op.finish is None])

            if remaining_time > max_remaining_time:
                max_remaining_time = remaining_time
                selected_job = job

    selected_op = None
    for op in selected_job.operation_queue:
        if op.finish is None:
            selected_op = op
            break

    # Find the first available machine to schedule the job
    selected_machine = select_machine(env.custom_scheduler.machines, selected_job, selected_op)

    return {
        'selected_machine': selected_machine,
        'selected_job': selected_job,
        'selected_op': selected_op,
    }

def LWKR_MOD(env):
    """
    Least Work Remaining + Modified Operational Due date
    - Pick the job with the minimum LWKR + MOD
    """
    min_lwkr_mod = np.inf
    selected_job = None
    jobs = env.custom_scheduler.jobs
    for job_list in jobs:
        for job in job_list:
            scheduled_time = 0
            remaining_time = 0
            current_time = 0
            for op in job.operation_queue:
                if op.finish is not None:
                    scheduled_time += op.duration
                else:
                    if current_time == 0:
                        current_time = op.duration
                    remaining_time += op.duration

            if remaining_time == 0:
                continue

            # Calculate the LWKR
            lwkr = remaining_time
            # Calculate the MOD
            total_time = scheduled_time + remaining_time
            r = 0  # arrival time
            c = job.deadline / total_time   # allowance factor (deadline / total processing time)
            mod = max(r + c * (scheduled_time + current_time), 0 + current_time)
            lwkr_mod = lwkr + mod

            if lwkr_mod < min_lwkr_mod:
                min_lwkr_mod = lwkr_mod
                selected_job = job

    selected_op = None
    for op in selected_job.operation_queue:
        if op.finish is None:
            selected_op = op
            break

    # Find the first available machine to schedule the job
    selected_machine = select_machine(env.custom_scheduler.machines, selected_job, selected_op)

    return {
        'selected_machine': selected_machine,
        'selected_job': selected_job,
        'selected_op': selected_op,
    }

def LWKR_SPT(env):
    """
    Least Work Remaining + Shortest Processing Time
    - Pick the job with the minimum LWKR + SPT
    """
    min_lwkr_spt = np.inf
    selected_job = None
    jobs = env.custom_scheduler.jobs
    for job_list in jobs:
        for job in job_list:
            remaining_time = 0
            current_time = 0
            for op in job.operation_queue:
                if op.finish is None:
                    if current_time == 0:
                        current_time = op.duration
                    remaining_time += op.duration

            if remaining_time == 0:
                continue

            # Calculate the LWKR
            lwkr = remaining_time
            # Calculate the SPT
            spt = current_time
            lwkr_spt = lwkr + spt

            if lwkr_spt < min_lwkr_spt:
                min_lwkr_spt = lwkr_spt
                selected_job = job

    selected_op = None
    for op in selected_job.operation_queue:
        if op.finish is None:
            selected_op = op
            break

    # Find the first available machine to schedule the job
    selected_machine = select_machine(env.custom_scheduler.machines, selected_job, selected_op)

    return {
        'selected_machine': selected_machine,
        'selected_job': selected_job,
        'selected_op': selected_op,
    }

def ETD(env):
    """
    Estimated Tardiness
    """
    max_etd = -np.inf
    selected_job = None
    jobs = env.custom_scheduler.jobs
    for job_list in jobs:
        for job in job_list:
            remaining_operations = [op for op in job.operation_queue if op.finish is None]

            if not remaining_operations:
                job.tardiness = job.operation_queue[-1].finish - job.deadline
                job.time_exceeded = max(0, job.operation_queue[-1].finish - job.deadline)
                job.estimated_tardiness = float(job.tardiness)
                job.is_done = True
                continue

            earliest_operation = remaining_operations[0]
            best_finish_times = [
                machine.cal_best_finish_time(op_earliest_start=earliest_operation.earliest_start,
                                             op_type=earliest_operation.type, op_duration=earliest_operation.duration)
                for machine in env.custom_scheduler.machines
            ]
            best_finish_times = [time for time in best_finish_times if time != -1]

            approx_best_finish_time = int(np.mean(best_finish_times))

            remaining_durations = [op.duration for op in remaining_operations[1:]]
            scaled_rate = (job.total_duration - sum(remaining_durations)) / job.total_duration
            # scaled_operation_deadline = scaled_rate * job.deadline
            # job.estimated_tardiness = approx_best_finish_time - scaled_operation_deadline
            tardiness = approx_best_finish_time - job.deadline
            job.estimated_tardiness = tardiness * scaled_rate

            if job.estimated_tardiness > max_etd:
                max_etd = job.estimated_tardiness
                selected_job = job

    selected_op = None
    for op in selected_job.operation_queue:
        if op.finish is None:
            selected_op = op
            break

    # Find the first available machine to schedule the job
    selected_machine = select_machine(env.custom_scheduler.machines, selected_job, selected_op)

    return {
        'selected_machine': selected_machine,
        'selected_job': selected_job,
        'selected_op': selected_op,
    }

def select_machine(machines, selected_job, selected_op):
    first_start_time = np.inf
    selected_machine = None
    for machine in machines:
        if not machine.can_process_operation(selected_op.type):
            continue

        start_time = find_first_start_time(machine, selected_job, selected_op)
        if start_time < first_start_time:
            first_start_time = start_time
            selected_machine = machine

    return selected_machine

def find_first_start_time(machine, job, operation):
    """
    Find the first available time to schedule the given operation on the given machine.

    :param machine: The machine to schedule the operation on.
    :param operation: The operation to schedule.
    :return: The first available time to schedule the operation on the machine.
    """

    operation_earliest_start = operation.earliest_start
    # Check for predecessor's finish time
    if operation.predecessor is not None:
        predecessor_operation = next(
            op for op in job.operation_queue if op.index == operation.predecessor
        )
        operation_earliest_start = max(operation_earliest_start, predecessor_operation.finish)

    operation_duration = operation.duration
    machine_operations = sorted(
        machine.operation_schedule, key=lambda operation: operation.start
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

    return min_earliest_start
### end : PDR 구현

### start : PDR / model 비교 실험 
def compare_pdr_model(pdrs, pdr_envs, model, model_envs, deterministic=True, sample_times=100, log_dir='./experiments/tmp/1', verbose=False, render=False):
    """
    Compare the given PDRs on the given environments.
    """
    writer = CustomLogger(log_dir=log_dir)

    if verbose:
        print(f'Comparing PDRs: {[pdr.__name__ for pdr in pdrs]}')

    # Log the results for each PDR
    for pdr in pdrs:
        results = eval_pdr(pdr, pdr_envs, render=render, verbose=verbose)

        for key, values in results.items():
            for i, value in enumerate(values):
                writer.add_scalar(f'{pdr.__name__}/{key}', value, i)

    if model is not None:
        results = eval_model(model, model_envs, writer, deterministic=deterministic,
                             sample_times=sample_times, verbose=verbose)

        for key, values in results.items():
            for i, value in enumerate(values):
                writer.add_scalar(f'PPO/{key}', value, i)

    writer.close()
    return
def eval_pdr(PDR, envs, render=False, verbose=False):

    """
    Evaluate the PDR on the given environments.

    :param PDR: The PDR function to evaluate.
    :param envs: The list of environments to evaluate on.
    :param render: Whether to render the environment.
    :return: The objective values of the environments. (makespan, total tardiness, idle time)
    """
    # ms_cost = envs[0].cost_makespan_per_time
    # tard_cost = envs[0].cost_deadline_per_time
    # pt_cst = envs[0].cost_processing_per_time
    # idle_cost = envs[0].cost_hole_per_time
    if verbose:
        print(f'Evaluating PDR: {PDR.__name__}')

    makespan_list = []
    tardiness_list = []
    processing_time_list = []
    idle_time_list = []

    makespan_cost_list = []
    tardiness_cost_list = []
    processing_time_cost_list = []
    idle_time_cost_list = []

    total_cost_list = []

    for env in envs:
        obs, info = env.reset()
        scheduler = env.custom_scheduler

        done = False
        while not done:
            action = PDR(env)
            scheduler.update_state(action)
            done = scheduler.is_done()

        obs = scheduler.get_observation()
        info = scheduler.get_info()

        makespan = scheduler._get_final_operation_finish()
        total_tardiness = sum(info['job_tardiness'])
        total_processing_time = info['sum_of_processing_time']
        total_idle_time = info['sum_of_hole_time']

        makespan_cost = info['cost_makespan']
        tardiness_cost = info['cost_deadline']
        processing_time_cost = info['cost_processing']
        idlet_time_cost = info['cost_hole']
        total_cost = info['total_cost']

        makespan_list.append(makespan)
        tardiness_list.append(total_tardiness)
        processing_time_list.append(total_processing_time)
        idle_time_list.append(total_idle_time)

        makespan_cost_list.append(makespan_cost)
        tardiness_cost_list.append(tardiness_cost)
        processing_time_cost_list.append(processing_time_cost)
        idle_time_cost_list.append(idlet_time_cost)
        total_cost_list.append(total_cost)

        if verbose:
            print(f'Repeat: {env.custom_scheduler.current_repeats}')
            print(
                f'makespan: {makespan}, total tardiness: {total_tardiness}, processing time: {total_processing_time}, idle time: {total_idle_time}')
            print(f'makespan cost: {makespan_cost}, tardiness cost: {tardiness_cost}, processing time cost: {processing_time_cost}, idle time cost: {idlet_time_cost}, total cost: {total_cost}')
        if render:
            env.render()
    return {
        'makespan': makespan_list,
        'total_tardiness': tardiness_list,
        'processing_time': processing_time_list,
        'idle_time': idle_time_list,
        'makespan_cost': makespan_cost_list,
        'tard_cost': tardiness_cost_list,
        'idle_time_cost': idle_time_cost_list,
        'total_cost': total_cost_list
    }
def eval_model(model, envs, writer, deterministic=True, sample_times=100, verbose=False, render=False):
    if verbose:
        print(f'Evaluating Model')

    makespan_list = []
    tardiness_list = []
    processing_time_list = []
    idle_time_list = []

    makespan_cost_list = []
    tardiness_cost_list = []
    processing_time_cost_list = []
    idle_time_cost_list = []

    total_cost_list = []

    for env in envs:
        if deterministic:
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
        else:
            # Find the best result in (sample_times) episodes
            best_result = None
            best_reward = -np.inf
            for _ in range(sample_times):
                obs, info = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=False, action_masks=env.action_masks())
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                if reward > best_reward:
                    best_result = info
                    best_reward = reward

            info = best_result

        makespan = info['makespan']
        total_tardiness = sum(info['job_tardiness'])
        total_processing_time = info['sum_of_processing_time']
        total_idle_time = info['sum_of_hole_time']

        makespan_cost = info['cost_makespan']
        tardiness_cost = info['cost_deadline']
        processing_time_cost = info['cost_processing']
        idlet_time_cost = info['cost_hole']
        total_cost = info['total_cost']

        makespan_list.append(makespan)
        tardiness_list.append(total_tardiness)
        processing_time_list.append(total_processing_time)
        idle_time_list.append(total_idle_time)

        makespan_cost_list.append(makespan_cost)
        tardiness_cost_list.append(tardiness_cost)
        processing_time_cost_list.append(processing_time_cost)
        idle_time_cost_list.append(idlet_time_cost)
        total_cost_list.append(total_cost)

        if verbose:
            print(f'Repeat: {env.custom_scheduler.current_repeats}')
            print(
                f'makespan: {makespan}, total tardiness: {total_tardiness}, processing time: {total_processing_time}, idle time: {total_idle_time}')
            print(f'makespan cost: {makespan_cost}, tardiness cost: {tardiness_cost}, processing time cost: {processing_time_cost}, idle time cost: {idlet_time_cost}, total cost: {total_cost}')
        if render:
            env.render()
    return {
        'makespan': makespan_list,
        'total_tardiness': tardiness_list,
        'processing_time': processing_time_list,
        'idle_time': idle_time_list,
        'makespan_cost': makespan_cost_list,
        'tard_cost': tardiness_cost_list,
        'idle_time_cost': idle_time_cost_list,
        'total_cost': total_cost_list
    }
### end : PDR / model 비교 실험 

### start : plot
def plot_pdr_comparison(pdrs, objs, with_model=True, log_dir='./experiments/tmp/1', pie=False, save_img=False):
    # Read the CSV file
    csv_file_path = os.path.join(log_dir, 'results.csv')
    df = pd.read_csv(csv_file_path)

    # Create a color map for algorithms
    algorithms = [pdr.__name__.replace('_over_', '/').replace('_', '+') for pdr in pdrs]
    if with_model:
        algorithms.append('PPO')

    # Create a color palette
    n_colors = len(algorithms)
    color_palette = plt.colormaps['Set2'](np.linspace(0, 1, n_colors))
    color_dict = dict(zip(algorithms, color_palette))

    # Objective charts
    for i, obj in enumerate(objs):
        fig, ax = plt.subplots(figsize=(8, 6))
        data = []
        for pdr in pdrs:
            pdr_name = pdr.__name__
            column_name = f'{pdr_name}/{obj}'
            pdr_data = df[column_name]
            pdr_name = pdr_name.replace('_over_', '/').replace('_', '+')
            pdr_df = pd.DataFrame({obj: pdr_data, 'Algo': pdr_name})
            data.append(pdr_df)
        if with_model:
            model_data = df[f'PPO/{obj}']
            model_df = pd.DataFrame({obj: model_data, 'Algo': 'PPO'})
            data.append(model_df)

        combined_data = pd.concat(data)

        # Use the color dictionary for the boxplot
        sns.boxplot(
            x='Algo',
            y=obj,
            data=combined_data,
            ax=ax,
            palette={algo: mcolors.rgb2hex(color_dict[algo]) for algo in color_dict}
        )

        # Calculate the mean values for each algorithm
        mean_values = combined_data.groupby('Algo')[obj].mean()

        # Find the lowest mean value
        lowest_mean_value = mean_values.min()

        # Draw a horizontal line at the lowest mean value
        ax.axhline(lowest_mean_value, color='red', linestyle='--',
                   linewidth=1, label=f'Lowest Mean: {lowest_mean_value:.2f}')

        ax.set_title(obj.replace('_', ' ').title())
        ax.set_ylabel(obj.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        # Save the plot as a PDF file
        if save_img:
            pdf_file_path = os.path.join(log_dir, f'{obj}_boxplot.pdf')
            plt.savefig(pdf_file_path, format='pdf')

        plt.show()

    # Win rate charts
    win_rate_objs = ['makespan_cost', 'tard_cost', 'idle_time_cost', 'total_cost']

    for i, obj in enumerate(win_rate_objs):
        fig, ax = plt.subplots(figsize=(10, 8))
        win_rates = {algo: 0 for algo in algorithms}

        best_algo_counts = df[[f'{pdr.__name__}/{obj}' for pdr in pdrs] +
                              ([f'PPO/{obj}'] if with_model else [])].idxmin(axis=1)
        for algo, pdr in zip(win_rates.keys(), pdrs):
            win_rates[algo] += (best_algo_counts == f'{pdr.__name__}/{obj}').sum()

        if with_model:
            win_rates['PPO'] += (best_algo_counts == f'PPO/{obj}').sum()

        total_instances = len(df)
        for algo in win_rates.keys():
            win_rates[algo] /= total_instances

        win_rate_df = pd.DataFrame(list(win_rates.items()), columns=['Algo', 'Win Rate'])
        print(win_rate_df)
        # assert sum of all win rates is 1 for each objective
        assert np.isclose(win_rate_df['Win Rate'].sum(), 1)

        if pie:
            win_rate_df = win_rate_df[win_rate_df['Win Rate'] > 0].sort_values('Win Rate', ascending=False)

            wedges, texts, autotexts = ax.pie(win_rate_df['Win Rate'],
                                              autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
                                              startangle=90,
                                              colors=[color_dict[algo] for algo in win_rate_df['Algo']],
                                              pctdistance=0.85)

            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig.gca().add_artist(centre_circle)
            ax.axis('equal')
            plt.setp(autotexts, size=10, weight="bold", color="black")
            ax.legend(wedges, win_rate_df['Algo'],
                      title="Algorithms",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1))
        else:
            sns.barplot(x='Algo', y='Win Rate', data=win_rate_df, ax=ax,
                        palette={algo: mcolors.rgb2hex(color_dict[algo]) for algo in color_dict})

        ax.set_title(f'Win Rate for {obj.replace("_", " ").title()}', fontsize=16)

        if not pie:
            ax.set_ylabel('Win Rate', fontsize=14)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        # Save the plot as a PDF file
        if save_img:
            pdf_file_path = os.path.join(log_dir, f'{obj}_winrate.pdf')
            plt.savefig(pdf_file_path, format='pdf')

        plt.show()

def plot_combined_comparison(pdrs, objs, with_model=True, log_dir='./experiments/tmp/1', pie=False, save_img=False):
    # Read the CSV file
    csv_file_path = os.path.join(log_dir, 'results.csv')
    df = pd.read_csv(csv_file_path)

    # Create a color map for algorithms
    algorithms = [pdr.__name__.replace('_over_', '/').replace('_', '+') for pdr in pdrs]
    if with_model:
        algorithms.append('PPO')
        
    color_names = [algo.replace('ETD', 'ETD/PDR').replace('PPO', 'Proposed') for algo in algorithms]

    # Create a color palette
    n_colors = len(color_names)
    color_palette = plt.colormaps['Set2'](np.linspace(0, 1, n_colors))
    color_dict = dict(zip(color_names, color_palette))

    # # Create a figure with 8 subplots (2 rows x 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.8}) #, constrained_layout=True)

    # Create a figure with 4 subplots (1 rows x 4 columns)
    # fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()

    # Objective charts
    for i, obj in enumerate(objs):
        ax = axes[i]
        data = []
        for pdr in pdrs:
            pdr_name = pdr.__name__
            column_name = f'{pdr_name}/{obj}'
            pdr_data = df[column_name]
            pdr_name = pdr_name.replace('_over_', '/').replace('_', '+')
            pdr_df = pd.DataFrame({obj: pdr_data, 'Algo': pdr_name})
            data.append(pdr_df)
        if with_model:
            model_data = df[f'PPO/{obj}']
            model_df = pd.DataFrame({obj: model_data, 'Algo': 'PPO'})
            data.append(model_df)

        combined_data = pd.concat(data)

        # Adjust algorithm names
        combined_data['Algo'] = combined_data['Algo'].replace({'ETD': 'ETD/PDR', 'PPO': 'Proposed'})

        # Use the color dictionary for the boxplot
        sns.boxplot(
            x='Algo',
            y=obj,
            data=combined_data,
            ax=ax,
            palette={algo: mcolors.rgb2hex(color_dict[algo]) for algo in color_dict}
        )

        # Calculate the mean values for each algorithm
        mean_values = combined_data.groupby('Algo')[obj].mean()
        print(mean_values)
        

        # # Find the lowest mean value
        # lowest_mean_value = mean_values.min()

        # # Draw a horizontal line at the lowest mean value
        # ax.axhline(lowest_mean_value, color='red', linestyle='--', linewidth=1, label=f'Lowest Mean: {lowest_mean_value:.2f}')
        title = obj.replace('_', ' ').title()
        if obj == 'tard_cost':
            title = "Tardiness Cost"
        ax.set_title(title)
        ax.set_xlabel('')  # Remove the 'Algo' label below x-axis labels
        # ylabel = obj.replace('_', ' ').title()
        # if obj == 'tard_cost':
        #     ylabel = 'Tardiness Cost'
        ylabel = "Cost"
        ax.set_ylabel(ylabel, labelpad=3)
        ax.tick_params(axis='x', rotation=45)
        # ax.legend()

    # Win rate charts
    win_rate_objs = ['makespan_cost', 'tard_cost', 'idle_time_cost', 'total_cost']

    for i, obj in enumerate(win_rate_objs):
        ax = axes[len(objs) + i]
        win_rates = {algo: 0 for algo in algorithms}

        best_algo_counts = df[[f'{pdr.__name__}/{obj}' for pdr in pdrs] +
                              ([f'PPO/{obj}'] if with_model else [])].idxmin(axis=1)
        for algo, pdr in zip(win_rates.keys(), pdrs):
            win_rates[algo] += (best_algo_counts == f'{pdr.__name__}/{obj}').sum()

        if with_model:
            win_rates['PPO'] += (best_algo_counts == f'PPO/{obj}').sum()

        total_instances = len(df)
        for algo in win_rates.keys():
            win_rates[algo] /= total_instances

        win_rate_df = pd.DataFrame(list(win_rates.items()), columns=['Algo', 'Win Rate'])
        win_rate_df['Algo'] = win_rate_df['Algo'].replace({'ETD': 'ETD/PDR', 'PPO': 'Proposed'})
        print(win_rate_df)
        # assert sum of all win rates is 1 for each objective
        assert np.isclose(win_rate_df['Win Rate'].sum(), 1)

        if pie:
            win_rate_df = win_rate_df[win_rate_df['Win Rate'] > 0].sort_values('Win Rate', ascending=False)

            wedges, texts, autotexts = ax.pie(win_rate_df['Win Rate'],
                                              autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
                                              startangle=90,
                                              colors=[color_dict[algo] for algo in win_rate_df['Algo']],
                                              pctdistance=0.85)

            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            ax.add_artist(centre_circle)
            ax.axis('equal')
            plt.setp(autotexts, size=10, weight="bold", color="black")
            # ax.legend(wedges, win_rate_df['Algo'],
            #           title="Algorithms",
            #           loc="center left",
            #           bbox_to_anchor=(1, 0, 0.5, 1))
            ax.legend(wedges, win_rate_df['Algo'],
                title="Algorithms",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.1),  # Adjust the y-coordinate to move the legend below the plot
                ncol=3)  # Adjust the number of columns in the legend as needed
        else:
            sns.barplot(x='Algo', y='Win Rate', data=win_rate_df, ax=ax,
                        palette={algo: mcolors.rgb2hex(color_dict[algo]) for algo in color_dict})

        title = obj.replace("_", " ").title()
        if obj == 'tard_cost':
            title = 'Tardiness Cost'
        ax.set_title(f'Win Rate on {title}', fontsize=15, pad=10) # Adjust title ~ plot gap 

        if not pie:
            ax.set_ylabel('Win Rate', fontsize=14)
            ax.tick_params(axis='x', rotation=45)

    # plt.tight_layout()

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.6)  # Adjust hspace value as needed

    # Save the combined plot as a PDF file
    if save_img:
        pdf_file_path = os.path.join(log_dir, 'combined_plots+win_rates_v2.pdf')
        plt.savefig(pdf_file_path, format='pdf', bbox_inches='tight')

    plt.show()
### end : plot

if __name__ == "__main__":
    # Random seed
    # np.random.seed(0)

    # n_jobs = 10
    # eval_instances = 100
    # mean, std = 3, 1

    # # Get 12 integers from Gaussian Distribution / mean : 3 std : 1
    # repeats = [[[max(1, int(np.random.normal(mean, std))), 1] for _ in range(n_jobs)] for _ in range(eval_instances)]
    # print(repeats)

    # repeats = [[[3] for _ in range(12)],
    #            [[4], [3], [3], [5], [4], [2], [3], [2], [2], [3], [3], [4]],
    #            [[3], [3], [3], [3], [4], [2], [3], [2], [1], [3], [3], [2]],
    #            [[5], [1], [3], [2], [4], [4], [3], [3], [2], [1], [2], [3]],
    #            [[4], [4], [2], [2], [1], [1], [1], [4], [2], [2], [1], [3]],
    #            [[1], [2], [2], [3], [2], [1], [2], [3], [3], [3], [2], [2]],
    #            [[2], [2], [2], [1], [3], [2], [1], [3], [2], [3], [3], [3]],
    #            [[4], [1], [3], [2], [2], [2], [2], [3], [1], [3], [3], [1]],
    #            [[4], [4], [4], [2], [1], [4], [2], [4], [3], [3], [3], [3]],
    #            [[3], [4], [3], [3], [4], [1], [1], [3], [1], [4], [2], [2]],
    #            [[4], [4], [4], [3], [2], [4], [2], [3], [3], [2], [3], [3]]]

    # repeats_12_10 = [[[3, 1] for _ in range(12)],
    #            [[4, 1], [3, 1], [3, 1], [5, 1], [4, 1], [2, 1], [3, 1], [2, 1], [2, 1], [3, 1], [3, 1], [4, 1]],
    #            [[3, 1], [3, 1], [3, 1], [3, 1], [4, 1], [2, 1], [3, 1], [2, 1], [1, 1], [3, 1], [3, 1], [2, 1]],
    #            [[5, 1], [1, 1], [3, 1], [2, 1], [4, 1], [4, 1], [3, 1], [3, 1], [2, 1], [1, 1], [2, 1], [3, 1]],
    #            [[4, 1], [4, 1], [2, 1], [2, 1], [1, 1], [1, 1], [1, 1], [4, 1], [2, 1], [2, 1], [1, 1], [3, 1]],
    #            [[1, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1]],
    #            [[2, 1], [2, 1], [2, 1], [1, 1], [3, 1], [2, 1], [1, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1]],
    #            [[4, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1], [1, 1], [3, 1], [3, 1], [1, 1]],
    #            [[4, 1], [4, 1], [4, 1], [2, 1], [1, 1], [4, 1], [2, 1], [4, 1], [3, 1], [3, 1], [3, 1], [3, 1]],
    #            [[3, 1], [4, 1], [3, 1], [3, 1], [4, 1], [1, 1], [1, 1], [3, 1], [1, 1], [4, 1], [2, 1], [2, 1]],
    #            [[4, 1], [4, 1], [4, 1], [3, 1], [2, 1], [4, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1]]]
    # repeats_12_100 = [[[4, 1], [3, 1], [3, 1], [5, 1], [4, 1], [2, 1], [3, 1], [2, 1], [2, 1], [3, 1], [3, 1], [4, 1]], [[3, 1], [3, 1], [3, 1], [3, 1], [4, 1], [2, 1], [3, 1], [2, 1], [1, 1], [3, 1], [3, 1], [2, 1]], [[5, 1], [1, 1], [3, 1], [2, 1], [4, 1], [4, 1], [3, 1], [3, 1], [2, 1], [1, 1], [2, 1], [3, 1]], [[4, 1], [4, 1], [2, 1], [2, 1], [1, 1], [1, 1], [1, 1], [4, 1], [2, 1], [2, 1], [1, 1], [3, 1]], [[1, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [2, 1], [1, 1], [3, 1], [2, 1], [1, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1]], [[4, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1], [1, 1], [3, 1], [3, 1], [1, 1]], [[4, 1], [4, 1], [4, 1], [2, 1], [1, 1], [4, 1], [2, 1], [4, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[3, 1], [4, 1], [3, 1], [3, 1], [4, 1], [1, 1], [1, 1], [3, 1], [1, 1], [4, 1], [2, 1], [2, 1]], [[4, 1], [4, 1], [4, 1], [3, 1], [2, 1], [4, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1]], [[3, 1], [1, 1], [3, 1], [4, 1], [2, 1], [2, 1], [2, 1], [4, 1], [3, 1], [3, 1], [2, 1], [3, 1]], [[2, 1], [3, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [1, 1], [1, 1], [3, 1], [3, 1], [3, 1]], [[5, 1], [3, 1], [2, 1], [4, 1], [1, 1], [2, 1], [2, 1], [4, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[4, 1], [1, 1], [1, 1], [2, 1], [2, 1], [4, 1], [3, 1], [3, 1], [1, 1], [3, 1], [1, 1], [1, 1]], [[4, 1], [3, 1], [3, 1], [3, 1], [3, 1], [2, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[2, 1], [1, 1], [2, 1], [1, 1], [3, 1], [1, 1], [1, 1], [3, 1], [2, 1], [4, 1], [1, 1], [3, 1]], [[2, 1], [1, 1], [3, 1], [2, 1], [3, 1], [3, 1], [5, 1], [4, 1], [2, 1], [2, 1], [4, 1], [3, 1]], [[3, 1], [1, 1], [2, 1], [2, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [3, 1], [5, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1], [1, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [2, 1], [1, 1], [2, 1], [2, 1], [3, 1], [1, 1], [3, 1], [4, 1], [1, 1], [3, 1], [3, 1]], [[2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1], [4, 1], [4, 1], [2, 1], [1, 1], [3, 1], [2, 1]], [[3, 1], [2, 1], [3, 1], [3, 1], [2, 1], [1, 1], [1, 1], [3, 1], [1, 1], [2, 1], [2, 1], [2, 1]], [[1, 1], [3, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [1, 1], [4, 1], [3, 1], [2, 1], [2, 1]], [[3, 1], [2, 1], [1, 1], [5, 1], [2, 1], [4, 1], [2, 1], [4, 1], [3, 1], [3, 1], [1, 1], [4, 1]], [[3, 1], [4, 1], [2, 1], [2, 1], [5, 1], [1, 1], [2, 1], [4, 1], [3, 1], [3, 1], [2, 1], [3, 1]], [[1, 1], [4, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [1, 1], [3, 1], [2, 1], [2, 1], [4, 1]], [[1, 1], [2, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1], [2, 1]], [[1, 1], [2, 1], [2, 1], [5, 1], [2, 1], [3, 1], [3, 1], [1, 1], [3, 1], [1, 1], [1, 1], [3, 1]], [[1, 1], [3, 1], [2, 1], [4, 1], [4, 1], [2, 1], [2, 1], [1, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [2, 1], [1, 1], [4, 1], [2, 1], [2, 1], [2, 1], [1, 1], [2, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [3, 1], [3, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [1, 1], [3, 1], [2, 1]], [[3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [1, 1], [2, 1], [3, 1], [1, 1], [3, 1], [1, 1], [1, 1]], [[3, 1], [1, 1], [2, 1], [1, 1], [4, 1], [3, 1], [3, 1], [2, 1], [2, 1], [1, 1], [2, 1], [2, 1]], [[3, 1], [4, 1], [4, 1], [3, 1], [2, 1], [1, 1], [3, 1], [3, 1], [1, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [2, 1], [1, 1], [3, 1], [2, 1], [2, 1], [4, 1], [3, 1], [3, 1], [3, 1], [4, 1], [4, 1]], [[3, 1], [3, 1], [1, 1], [3, 1], [2, 1], [3, 1], [3, 1], [1, 1], [2, 1], [3, 1], [2, 1], [1, 1]], [[3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [2, 1], [1, 1], [2, 1], [3, 1]], [[3, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [1, 1], [3, 1], [3, 1], [3, 1], [3, 1], [5, 1]], [[2, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1], [2, 1], [5, 1], [2, 1], [2, 1]], [[1, 1], [3, 1], [1, 1], [2, 1], [2, 1], [2, 1], [4, 1], [3, 1], [4, 1], [2, 1], [4, 1], [3, 1]], [[3, 1], [4, 1], [2, 1], [2, 1], [3, 1], [1, 1], [4, 1], [3, 1], [2, 1], [3, 1], [1, 1], [2, 1]], [[4, 1], [4, 1], [5, 1], [2, 1], [2, 1], [2, 1], [1, 1], [2, 1], [3, 1], [2, 1], [4, 1], [2, 1]], [[2, 1], [2, 1], [1, 1], [1, 1], [1, 1], [4, 1], [3, 1], [4, 1], [1, 1], [1, 1], [2, 1], [3, 1]], [[3, 1], [4, 1], [3, 1], [4, 1], [2, 1], [2, 1], [1, 1], [2, 1], [3, 1], [2, 1], [3, 1], [4, 1]], [[1, 1], [3, 1], [1, 1], [3, 1], [4, 1], [3, 1], [1, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[3, 1], [1, 1], [2, 1], [4, 1], [2, 1], [2, 1], [4, 1], [1, 1], [4, 1], [2, 1], [3, 1], [3, 1]], [[3, 1], [2, 1], [2, 1], [4, 1], [1, 1], [3, 1], [1, 1], [2, 1], [5, 1], [2, 1], [3, 1], [1, 1]], [[2, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [1, 1], [3, 1], [3, 1], [3, 1], [2, 1]], [[2, 1], [1, 1], [2, 1], [4, 1], [1, 1], [2, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[2, 1], [1, 1], [3, 1], [3, 1], [2, 1], [1, 1], [3, 1], [3, 1], [2, 1], [2, 1], [2, 1], [
    #     2, 1]], [[1, 1], [3, 1], [2, 1], [3, 1], [1, 1], [2, 1], [2, 1], [1, 1], [3, 1], [2, 1], [1, 1], [1, 1]], [[3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [4, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [1, 1], [3, 1], [4, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1], [2, 1], [2, 1], [3, 1]], [[2, 1], [3, 1], [3, 1], [4, 1], [2, 1], [4, 1], [4, 1], [2, 1], [1, 1], [2, 1], [4, 1], [2, 1]], [[2, 1], [4, 1], [3, 1], [1, 1], [3, 1], [3, 1], [3, 1], [3, 1], [2, 1], [1, 1], [2, 1], [2, 1]], [[2, 1], [1, 1], [3, 1], [3, 1], [2, 1], [2, 1], [2, 1], [4, 1], [1, 1], [3, 1], [2, 1], [1, 1]], [[1, 1], [2, 1], [3, 1], [3, 1], [2, 1], [4, 1], [4, 1], [3, 1], [2, 1], [4, 1], [3, 1], [3, 1]], [[2, 1], [1, 1], [5, 1], [1, 1], [2, 1], [5, 1], [3, 1], [3, 1], [1, 1], [1, 1], [2, 1], [4, 1]], [[2, 1], [2, 1], [3, 1], [4, 1], [1, 1], [1, 1], [2, 1], [4, 1], [3, 1], [5, 1], [3, 1], [2, 1]], [[4, 1], [3, 1], [2, 1], [3, 1], [2, 1], [2, 1], [4, 1], [2, 1], [2, 1], [2, 1], [4, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [2, 1], [4, 1], [3, 1], [3, 1], [2, 1], [2, 1]], [[2, 1], [1, 1], [5, 1], [1, 1], [3, 1], [1, 1], [2, 1], [2, 1], [1, 1], [1, 1], [4, 1], [2, 1]], [[2, 1], [3, 1], [3, 1], [3, 1], [1, 1], [3, 1], [3, 1], [1, 1], [3, 1], [4, 1], [2, 1], [3, 1]], [[3, 1], [2, 1], [3, 1], [1, 1], [2, 1], [3, 1], [3, 1], [3, 1], [3, 1], [1, 1], [2, 1], [3, 1]], [[4, 1], [2, 1], [2, 1], [3, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [2, 1], [1, 1], [2, 1]], [[4, 1], [3, 1], [3, 1], [1, 1], [3, 1], [3, 1], [3, 1], [3, 1], [1, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [4, 1], [2, 1], [2, 1], [3, 1], [1, 1], [2, 1], [2, 1], [4, 1], [3, 1], [2, 1], [2, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [1, 1], [2, 1], [2, 1], [3, 1], [4, 1], [2, 1], [3, 1], [4, 1]], [[2, 1], [4, 1], [2, 1], [3, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [3, 1], [3, 1], [5, 1]], [[4, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [2, 1], [2, 1], [3, 1], [3, 1]], [[3, 1], [2, 1], [3, 1], [5, 1], [3, 1], [2, 1], [3, 1], [2, 1], [2, 1], [3, 1], [1, 1], [1, 1]], [[2, 1], [1, 1], [1, 1], [2, 1], [1, 1], [2, 1], [2, 1], [3, 1], [4, 1], [3, 1], [3, 1], [3, 1]], [[3, 1], [1, 1], [2, 1], [3, 1], [1, 1], [3, 1], [4, 1], [2, 1], [2, 1], [2, 1], [3, 1], [1, 1]], [[2, 1], [2, 1], [3, 1], [1, 1], [4, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [3, 1], [2, 1]], [[3, 1], [2, 1], [2, 1], [3, 1], [3, 1], [2, 1], [1, 1], [4, 1], [2, 1], [1, 1], [5, 1], [2, 1]], [[1, 1], [2, 1], [3, 1], [3, 1], [2, 1], [4, 1], [2, 1], [1, 1], [3, 1], [4, 1], [2, 1], [1, 1]], [[4, 1], [1, 1], [1, 1], [3, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [4, 1], [4, 1], [3, 1]], [[3, 1], [5, 1], [2, 1], [2, 1], [3, 1], [4, 1], [4, 1], [2, 1], [3, 1], [1, 1], [2, 1], [2, 1]], [[1, 1], [4, 1], [2, 1], [2, 1], [3, 1], [4, 1], [3, 1], [5, 1], [2, 1], [3, 1], [3, 1], [4, 1]], [[3, 1], [2, 1], [2, 1], [3, 1], [4, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [5, 1], [1, 1]], [[2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1], [3, 1], [4, 1], [3, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [4, 1], [5, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1], [2, 1], [2, 1], [4, 1]], [[2, 1], [3, 1], [2, 1], [3, 1], [4, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [3, 1], [1, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [3, 1], [1, 1]], [[2, 1], [3, 1], [1, 1], [4, 1], [3, 1], [4, 1], [2, 1], [2, 1], [3, 1], [1, 1], [3, 1], [3, 1]], [[4, 1], [3, 1], [2, 1], [1, 1], [4, 1], [2, 1], [1, 1], [1, 1], [4, 1], [2, 1], [2, 1], [3, 1]], [[2, 1], [2, 1], [3, 1], [4, 1], [3, 1], [3, 1], [3, 1], [1, 1], [4, 1], [3, 1], [3, 1], [4, 1]], [[4, 1], [3, 1], [3, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [4, 1], [6, 1], [3, 1]], [[1, 1], [4, 1], [2, 1], [2, 1], [3, 1], [1, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[3, 1], [3, 1], [1, 1], [1, 1], [2, 1], [1, 1], [3, 1], [3, 1], [3, 1], [1, 1], [2, 1], [3, 1]], [[2, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [1, 1], [2, 1], [3, 1], [3, 1], [2, 1], [2, 1]], [[1, 1], [2, 1], [5, 1], [2, 1], [3, 1], [2, 1], [3, 1], [3, 1], [1, 1], [2, 1], [4, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [3, 1], [4, 1], [3, 1], [2, 1], [4, 1], [3, 1], [1, 1], [5, 1], [4, 1]], [[2, 1], [3, 1], [4, 1], [2, 1], [3, 1], [4, 1], [1, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1]], [[2, 1], [1, 1], [1, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1], [3, 1], [3, 1], [4, 1], [3, 1]], [[1, 1], [1, 1], [4, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [4, 1], [2, 1], [2, 1], [1, 1]], [[3, 1], [3, 1], [3, 1], [4, 1], [3, 1], [3, 1], [4, 1], [1, 1], [2, 1], [4, 1], [3, 1], [3, 1]], [[3, 1], [3, 1], [4, 1], [3, 1], [3, 1], [3, 1], [3, 1], [1, 1], [3, 1], [2, 1], [4, 1], [2, 1]], [[3, 1], [3, 1], [2, 1], [3, 1], [4, 1], [2, 1], [2, 1], [2, 1], [1, 1], [2, 1], [3, 1], [4, 1]], [[3, 1], [2, 1], [2, 1], [4, 1], [2, 1], [2, 1], [4, 1], [4, 1], [3, 1], [2, 1], [1, 1], [3, 1]]]

    # repeats_5_100 = [[[4, 1], [3, 1], [3, 1], [5, 1], [4, 1]], [[2, 1], [3, 1], [2, 1], [2, 1], [3, 1]], [[3, 1], [4, 1], [3, 1], [3, 1], [3, 1]], [[3, 1], [4, 1], [2, 1], [3, 1], [2, 1]], [[1, 1], [3, 1], [3, 1], [2, 1], [5, 1]], [[1, 1], [3, 1], [2, 1], [4, 1], [4, 1]], [[3, 1], [3, 1], [2, 1], [1, 1], [2, 1]], [[3, 1], [4, 1], [4, 1], [2, 1], [2, 1]], [[1, 1], [1, 1], [1, 1], [4, 1], [2, 1]], [[2, 1], [1, 1], [3, 1], [1, 1], [2, 1]], [[2, 1], [3, 1], [2, 1], [1, 1], [2, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [2, 1], [1, 1], [3, 1]], [[2, 1], [1, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [4, 1], [1, 1], [3, 1]], [[2, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[1, 1], [3, 1], [3, 1], [1, 1], [4, 1]], [[4, 1], [4, 1], [2, 1], [1, 1], [4, 1]], [[2, 1], [4, 1], [3, 1], [3, 1], [3, 1]], [[3, 1], [3, 1], [4, 1], [3, 1], [3, 1]], [[4, 1], [1, 1], [1, 1], [3, 1], [1, 1]], [[4, 1], [2, 1], [2, 1], [4, 1], [4, 1]], [[4, 1], [3, 1], [2, 1], [4, 1], [2, 1]], [[3, 1], [3, 1], [2, 1], [3, 1], [3, 1]], [[3, 1], [1, 1], [3, 1], [4, 1], [2, 1]], [[2, 1], [2, 1], [4, 1], [3, 1], [3, 1]], [[2, 1], [3, 1], [2, 1], [3, 1], [2, 1]], [[3, 1], [3, 1], [2, 1], [3, 1], [1, 1]], [[1, 1], [3, 1], [3, 1], [3, 1], [5, 1]], [[3, 1], [2, 1], [4, 1], [1, 1], [2, 1]], [[2, 1], [4, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [4, 1], [1, 1], [1, 1], [2, 1]], [[2, 1], [4, 1], [3, 1], [3, 1], [1, 1]], [[3, 1], [1, 1], [1, 1], [4, 1], [3, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [1, 1]], [[3, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[2, 1], [1, 1], [2, 1], [1, 1], [3, 1]], [[1, 1], [1, 1], [3, 1], [2, 1], [4, 1]], [[1, 1], [3, 1], [2, 1], [1, 1], [3, 1]], [[2, 1], [3, 1], [3, 1], [5, 1], [4, 1]], [[2, 1], [2, 1], [4, 1], [3, 1], [3, 1]], [[1, 1], [2, 1], [2, 1], [3, 1], [2, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [5, 1], [2, 1]], [[2, 1], [2, 1], [2, 1], [3, 1], [1, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [2, 1]], [[1, 1], [2, 1], [2, 1], [3, 1], [1, 1]], [[3, 1], [4, 1], [1, 1], [3, 1], [3, 1]], [[2, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[1, 1], [4, 1], [4, 1], [2, 1], [
    #     1, 1]], [[3, 1], [2, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [2, 1], [1, 1], [1, 1], [3, 1]], [[1, 1], [2, 1], [2, 1], [2, 1], [1, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [1, 1], [4, 1], [3, 1], [2, 1]], [[2, 1], [3, 1], [2, 1], [1, 1], [5, 1]], [[2, 1], [4, 1], [2, 1], [4, 1], [3, 1]], [[3, 1], [1, 1], [4, 1], [3, 1], [4, 1]], [[2, 1], [2, 1], [5, 1], [1, 1], [2, 1]], [[4, 1], [3, 1], [3, 1], [2, 1], [3, 1]], [[1, 1], [4, 1], [2, 1], [2, 1], [3, 1]], [[2, 1], [1, 1], [1, 1], [3, 1], [2, 1]], [[2, 1], [4, 1], [1, 1], [2, 1], [2, 1]], [[3, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [2, 1], [1, 1]], [[2, 1], [2, 1], [5, 1], [2, 1], [3, 1]], [[3, 1], [1, 1], [3, 1], [1, 1], [1, 1]], [[3, 1], [1, 1], [3, 1], [2, 1], [4, 1]], [[4, 1], [2, 1], [2, 1], [1, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [1, 1], [4, 1], [2, 1], [2, 1]], [[2, 1], [1, 1], [2, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [3, 1], [3, 1], [2, 1]], [[3, 1], [2, 1], [2, 1], [2, 1], [1, 1]], [[3, 1], [2, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [1, 1], [2, 1], [3, 1]], [[1, 1], [3, 1], [1, 1], [1, 1], [3, 1]], [[1, 1], [2, 1], [1, 1], [4, 1], [3, 1]], [[3, 1], [2, 1], [2, 1], [1, 1], [2, 1]], [[2, 1], [3, 1], [4, 1], [4, 1], [3, 1]], [[2, 1], [1, 1], [3, 1], [3, 1], [1, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [2, 1]], [[1, 1], [3, 1], [2, 1], [2, 1], [4, 1]], [[3, 1], [3, 1], [3, 1], [4, 1], [4, 1]], [[3, 1], [3, 1], [1, 1], [3, 1], [2, 1]], [[3, 1], [3, 1], [1, 1], [2, 1], [3, 1]], [[2, 1], [1, 1], [3, 1], [3, 1], [2, 1]], [[3, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [1, 1], [2, 1], [3, 1], [3, 1]], [[2, 1], [3, 1], [2, 1], [2, 1], [2, 1]], [[1, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[5, 1], [2, 1], [1, 1], [3, 1], [2, 1]], [[2, 1], [2, 1], [2, 1], [3, 1], [2, 1]], [[5, 1], [2, 1], [2, 1], [1, 1], [3, 1]], [[1, 1], [2, 1], [2, 1], [2, 1], [4, 1]], [[3, 1], [4, 1], [2, 1], [4, 1], [3, 1]], [[3, 1], [4, 1], [2, 1], [2, 1], [3, 1]], [[1, 1], [4, 1], [3, 1], [2, 1], [3, 1]], [[1, 1], [2, 1], [4, 1], [4, 1], [5, 1]], [[2, 1], [2, 1], [2, 1], [1, 1], [2, 1]]]

    # repeats_7_100 = [[[4, 1], [3, 1], [3, 1], [5, 1], [4, 1], [2, 1], [3, 1]], [[2, 1], [2, 1], [3, 1], [3, 1], [4, 1], [3, 1], [3, 1]], [[3, 1], [3, 1], [4, 1], [2, 1], [3, 1], [2, 1], [1, 1]], [[3, 1], [3, 1], [2, 1], [5, 1], [1, 1], [3, 1], [2, 1]], [[4, 1], [4, 1], [3, 1], [3, 1], [2, 1], [1, 1], [2, 1]], [[3, 1], [4, 1], [4, 1], [2, 1], [2, 1], [1, 1], [1, 1]], [[1, 1], [4, 1], [2, 1], [2, 1], [1, 1], [3, 1], [1, 1]], [[2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[1, 1], [3, 1], [2, 1], [1, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [4, 1], [1, 1], [3, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [1, 1], [3, 1], [3, 1], [1, 1]], [[4, 1], [4, 1], [4, 1], [2, 1], [1, 1], [4, 1], [2, 1]], [[4, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [4, 1]], [[3, 1], [3, 1], [4, 1], [1, 1], [1, 1], [3, 1], [1, 1]], [[4, 1], [2, 1], [2, 1], [4, 1], [4, 1], [4, 1], [3, 1]], [[2, 1], [4, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [1, 1], [3, 1], [4, 1], [2, 1], [2, 1]], [[2, 1], [4, 1], [3, 1], [3, 1], [2, 1], [3, 1], [2, 1]], [[3, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [1, 1]], [[1, 1], [3, 1], [3, 1], [3, 1], [5, 1], [3, 1], [2, 1]], [[4, 1], [1, 1], [2, 1], [2, 1], [4, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [4, 1], [1, 1], [1, 1], [2, 1], [2, 1]], [[4, 1], [3, 1], [3, 1], [1, 1], [3, 1], [1, 1], [1, 1]], [[4, 1], [3, 1], [3, 1], [3, 1], [3, 1], [2, 1], [1, 1]], [[3, 1], [2, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1]], [[2, 1], [1, 1], [3, 1], [1, 1], [1, 1], [3, 1], [2, 1]], [[4, 1], [1, 1], [3, 1], [2, 1], [1, 1], [3, 1], [2, 1]], [[3, 1], [3, 1], [5, 1], [4, 1], [2, 1], [2, 1], [4, 1]], [[3, 1], [3, 1], [1, 1], [2, 1], [2, 1], [3, 1], [2, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[3, 1], [5, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[1, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1], [1, 1]], [[2, 1], [2, 1], [3, 1], [1, 1], [3, 1], [4, 1], [1, 1]], [[3, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[1, 1], [4, 1], [4, 1], [2, 1], [1, 1], [3, 1], [2, 1]], [[3, 1], [2, 1], [3, 1], [3, 1], [2, 1], [1, 1], [1, 1]], [[3, 1], [1, 1], [2, 1], [2, 1], [2, 1], [1, 1], [3, 1]], [[3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [1, 1], [4, 1]], [[3, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [5, 1]], [[2, 1], [4, 1], [2, 1], [4, 1], [3, 1], [3, 1], [1, 1]], [[4, 1], [3, 1], [4, 1], [2, 1], [2, 1], [5, 1], [1, 1]], [[2, 1], [4, 1], [3, 1], [3, 1], [2, 1], [3, 1], [1, 1]], [[4, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [1, 1]], [[3, 1], [2, 1], [2, 1], [4, 1], [1, 1], [2, 1], [2, 1]], [[3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[3, 1], [2, 1], [1, 1], [2, 1], [2, 1], [5, 1], [2, 1]], [[3, 1], [3, 1], [1, 1], [3, 1], [1, 1], [1, 1], [3, 1]], [[1, 1], [3, 1], [2, 1], [4, 1], [4, 1], [2, 1], [2, 1]], [[1, 1], [2, 1], [2, 1], [2, 1], [3, 1], [3, 1], [
    #     3, 1]], [[2, 1], [1, 1], [4, 1], [2, 1], [2, 1], [2, 1], [1, 1]], [[2, 1], [2, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [1, 1], [3, 1]], [[2, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [1, 1]], [[2, 1], [3, 1], [1, 1], [3, 1], [1, 1], [1, 1], [3, 1]], [[1, 1], [2, 1], [1, 1], [4, 1], [3, 1], [3, 1], [2, 1]], [[2, 1], [1, 1], [2, 1], [2, 1], [3, 1], [4, 1], [4, 1]], [[3, 1], [2, 1], [1, 1], [3, 1], [3, 1], [1, 1], [3, 1]], [[3, 1], [3, 1], [2, 1], [2, 1], [1, 1], [3, 1], [2, 1]], [[2, 1], [4, 1], [3, 1], [3, 1], [3, 1], [4, 1], [4, 1]], [[3, 1], [3, 1], [1, 1], [3, 1], [2, 1], [3, 1], [3, 1]], [[1, 1], [2, 1], [3, 1], [2, 1], [1, 1], [3, 1], [3, 1]], [[2, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [2, 1]], [[1, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [2, 1]], [[2, 1], [2, 1], [1, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[5, 1], [2, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [3, 1], [2, 1], [5, 1], [2, 1], [2, 1], [1, 1]], [[3, 1], [1, 1], [2, 1], [2, 1], [2, 1], [4, 1], [3, 1]], [[4, 1], [2, 1], [4, 1], [3, 1], [3, 1], [4, 1], [2, 1]], [[2, 1], [3, 1], [1, 1], [4, 1], [3, 1], [2, 1], [3, 1]], [[1, 1], [2, 1], [4, 1], [4, 1], [5, 1], [2, 1], [2, 1]], [[2, 1], [1, 1], [2, 1], [3, 1], [2, 1], [4, 1], [2, 1]], [[2, 1], [2, 1], [1, 1], [1, 1], [1, 1], [4, 1], [3, 1]], [[4, 1], [1, 1], [1, 1], [2, 1], [3, 1], [3, 1], [4, 1]], [[3, 1], [4, 1], [2, 1], [2, 1], [1, 1], [2, 1], [3, 1]], [[2, 1], [3, 1], [4, 1], [1, 1], [3, 1], [1, 1], [3, 1]], [[4, 1], [3, 1], [1, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [3, 1], [1, 1], [2, 1], [4, 1], [2, 1], [2, 1]], [[4, 1], [1, 1], [4, 1], [2, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [2, 1], [4, 1], [1, 1], [3, 1], [1, 1], [2, 1]], [[5, 1], [2, 1], [3, 1], [1, 1], [2, 1], [3, 1], [3, 1]], [[3, 1], [3, 1], [3, 1], [3, 1], [1, 1], [3, 1], [3, 1]], [[3, 1], [2, 1], [2, 1], [1, 1], [2, 1], [4, 1], [1, 1]], [[2, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[2, 1], [1, 1], [3, 1], [3, 1], [2, 1], [1, 1], [3, 1]], [[3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1], [3, 1]], [[2, 1], [3, 1], [1, 1], [2, 1], [2, 1], [1, 1], [3, 1]], [[2, 1], [1, 1], [1, 1], [3, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [4, 1], [1, 1], [3, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [1, 1], [3, 1], [4, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [2, 1], [2, 1], [3, 1], [2, 1]], [[3, 1], [3, 1], [4, 1], [2, 1], [4, 1], [4, 1], [2, 1]], [[1, 1], [2, 1], [4, 1], [2, 1], [2, 1], [4, 1], [3, 1]], [[1, 1], [3, 1], [3, 1], [3, 1], [3, 1], [2, 1], [1, 1]], [[2, 1], [2, 1], [2, 1], [1, 1], [3, 1], [3, 1], [2, 1]], [[2, 1], [2, 1], [4, 1], [1, 1], [3, 1], [2, 1], [1, 1]], [[1, 1], [2, 1], [3, 1], [3, 1], [2, 1], [4, 1], [4, 1]], [[3, 1], [2, 1], [4, 1], [3, 1], [3, 1], [2, 1], [1, 1]], [[5, 1], [1, 1], [2, 1], [5, 1], [3, 1], [3, 1], [1, 1]], [[1, 1], [2, 1], [4, 1], [2, 1], [2, 1], [3, 1], [4, 1]]]
    # repeats_8_100 = [[[4, 1], [3, 1], [3, 1], [5, 1], [4, 1], [2, 1], [3, 1], [2, 1]], [[2, 1], [3, 1], [3, 1], [4, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[4, 1], [2, 1], [3, 1], [2, 1], [1, 1], [3, 1], [3, 1], [2, 1]], [[5, 1], [1, 1], [3, 1], [2, 1], [4, 1], [4, 1], [3, 1], [3, 1]], [[2, 1], [1, 1], [2, 1], [3, 1], [4, 1], [4, 1], [2, 1], [2, 1]], [[1, 1], [1, 1], [1, 1], [4, 1], [2, 1], [2, 1], [1, 1], [3, 1]], [[1, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1]], [[3, 1], [2, 1], [1, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1]], [[4, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[1, 1], [3, 1], [3, 1], [1, 1], [4, 1], [4, 1], [4, 1], [2, 1]], [[1, 1], [4, 1], [2, 1], [4, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[3, 1], [4, 1], [3, 1], [3, 1], [4, 1], [1, 1], [1, 1], [3, 1]], [[1, 1], [4, 1], [2, 1], [2, 1], [4, 1], [4, 1], [4, 1], [3, 1]], [[2, 1], [4, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1]], [[3, 1], [1, 1], [3, 1], [4, 1], [2, 1], [2, 1], [2, 1], [4, 1]], [[3, 1], [3, 1], [2, 1], [3, 1], [2, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [2, 1], [3, 1], [1, 1], [1, 1], [3, 1], [3, 1], [3, 1]], [[5, 1], [3, 1], [2, 1], [4, 1], [1, 1], [2, 1], [2, 1], [4, 1]], [[2, 1], [2, 1], [2, 1], [2, 1], [4, 1], [1, 1], [1, 1], [2, 1]], [[2, 1], [4, 1], [3, 1], [3, 1], [1, 1], [3, 1], [1, 1], [1, 1]], [[4, 1], [3, 1], [3, 1], [3, 1], [3, 1], [2, 1], [1, 1], [3, 1]], [[2, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [1, 1]], [[3, 1], [1, 1], [1, 1], [3, 1], [2, 1], [4, 1], [1, 1], [3, 1]], [[2, 1], [1, 1], [3, 1], [2, 1], [3, 1], [3, 1], [5, 1], [4, 1]], [[2, 1], [2, 1], [4, 1], [3, 1], [3, 1], [1, 1], [2, 1], [2, 1]], [[3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [3, 1], [5, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[1, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1], [1, 1], [2, 1]], [[2, 1], [3, 1], [1, 1], [3, 1], [4, 1], [1, 1], [3, 1], [3, 1]], [[2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1], [4, 1], [4, 1]], [[2, 1], [1, 1], [3, 1], [2, 1], [3, 1], [2, 1], [3, 1], [3, 1]], [[2, 1], [1, 1], [1, 1], [3, 1], [1, 1], [2, 1], [2, 1], [2, 1]], [[1, 1], [3, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [1, 1]], [[4, 1], [3, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [5, 1]], [[2, 1], [4, 1], [2, 1], [4, 1], [3, 1], [3, 1], [1, 1], [4, 1]], [[3, 1], [4, 1], [2, 1], [2, 1], [5, 1], [1, 1], [2, 1], [4, 1]], [[3, 1], [3, 1], [2, 1], [3, 1], [1, 1], [4, 1], [2, 1], [2, 1]], [[3, 1], [2, 1], [1, 1], [1, 1], [3, 1], [2, 1], [2, 1], [4, 1]], [[1, 1], [2, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [2, 1], [5, 1]], [[2, 1], [3, 1], [3, 1], [1, 1], [3, 1], [1, 1], [1, 1], [3, 1]], [[1, 1], [3, 1], [2, 1], [4, 1], [4, 1], [2, 1], [2, 1], [1, 1]], [[2, 1], [2, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [1, 1]], [[4, 1], [2, 1], [2, 1], [2, 1], [1, 1], [2, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [3, 1], [3, 1], [2, 1], [3, 1], [2, 1], [2, 1]], [[2, 1], [1, 1], [3, 1], [2, 1], [3, 1], [2, 1], [3, 1], [3, 1]], [[3, 1], [1, 1], [2, 1], [3, 1], [1, 1], [3, 1], [1, 1], [1, 1]], [[3, 1], [1, 1], [2, 1], [1, 1], [4, 1], [3, 1], [3, 1], [2, 1]], [[2, 1], [1, 1], [2, 1], [2, 1], [3, 1], [4, 1], [4, 1], [
    #     3, 1]], [[2, 1], [1, 1], [3, 1], [3, 1], [1, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [2, 1], [1, 1], [3, 1], [2, 1], [2, 1], [4, 1], [3, 1]], [[3, 1], [3, 1], [4, 1], [4, 1], [3, 1], [3, 1], [1, 1], [3, 1]], [[2, 1], [3, 1], [3, 1], [1, 1], [2, 1], [3, 1], [2, 1], [1, 1]], [[3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [1, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [2, 1]], [[2, 1], [2, 1], [1, 1], [3, 1], [3, 1], [3, 1], [3, 1], [5, 1]], [[2, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[2, 1], [5, 1], [2, 1], [2, 1], [1, 1], [3, 1], [1, 1], [2, 1]], [[2, 1], [2, 1], [4, 1], [3, 1], [4, 1], [2, 1], [4, 1], [3, 1]], [[3, 1], [4, 1], [2, 1], [2, 1], [3, 1], [1, 1], [4, 1], [3, 1]], [[2, 1], [3, 1], [1, 1], [2, 1], [4, 1], [4, 1], [5, 1], [2, 1]], [[2, 1], [2, 1], [1, 1], [2, 1], [3, 1], [2, 1], [4, 1], [2, 1]], [[2, 1], [2, 1], [1, 1], [1, 1], [1, 1], [4, 1], [3, 1], [4, 1]], [[1, 1], [1, 1], [2, 1], [3, 1], [3, 1], [4, 1], [3, 1], [4, 1]], [[2, 1], [2, 1], [1, 1], [2, 1], [3, 1], [2, 1], [3, 1], [4, 1]], [[1, 1], [3, 1], [1, 1], [3, 1], [4, 1], [3, 1], [1, 1], [2, 1]], [[2, 1], [2, 1], [2, 1], [2, 1], [3, 1], [1, 1], [2, 1], [4, 1]], [[2, 1], [2, 1], [4, 1], [1, 1], [4, 1], [2, 1], [3, 1], [3, 1]], [[3, 1], [2, 1], [2, 1], [4, 1], [1, 1], [3, 1], [1, 1], [2, 1]], [[5, 1], [2, 1], [3, 1], [1, 1], [2, 1], [3, 1], [3, 1], [3, 1]], [[3, 1], [3, 1], [3, 1], [1, 1], [3, 1], [3, 1], [3, 1], [2, 1]], [[2, 1], [1, 1], [2, 1], [4, 1], [1, 1], [2, 1], [1, 1], [3, 1]], [[2, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [3, 1], [3, 1]], [[2, 1], [1, 1], [3, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[1, 1], [3, 1], [2, 1], [3, 1], [1, 1], [2, 1], [2, 1], [1, 1]], [[3, 1], [2, 1], [1, 1], [1, 1], [3, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [4, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [1, 1], [3, 1], [4, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[3, 1], [2, 1], [2, 1], [3, 1], [2, 1], [3, 1], [3, 1], [4, 1]], [[2, 1], [4, 1], [4, 1], [2, 1], [1, 1], [2, 1], [4, 1], [2, 1]], [[2, 1], [4, 1], [3, 1], [1, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [1, 1], [2, 1], [2, 1], [2, 1], [1, 1], [3, 1], [3, 1]], [[2, 1], [2, 1], [2, 1], [4, 1], [1, 1], [3, 1], [2, 1], [1, 1]], [[1, 1], [2, 1], [3, 1], [3, 1], [2, 1], [4, 1], [4, 1], [3, 1]], [[2, 1], [4, 1], [3, 1], [3, 1], [2, 1], [1, 1], [5, 1], [1, 1]], [[2, 1], [5, 1], [3, 1], [3, 1], [1, 1], [1, 1], [2, 1], [4, 1]], [[2, 1], [2, 1], [3, 1], [4, 1], [1, 1], [1, 1], [2, 1], [4, 1]], [[3, 1], [5, 1], [3, 1], [2, 1], [4, 1], [3, 1], [2, 1], [3, 1]], [[2, 1], [2, 1], [4, 1], [2, 1], [2, 1], [2, 1], [4, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [2, 1], [4, 1]], [[3, 1], [3, 1], [2, 1], [2, 1], [2, 1], [1, 1], [5, 1], [1, 1]], [[3, 1], [1, 1], [2, 1], [2, 1], [1, 1], [1, 1], [4, 1], [2, 1]], [[2, 1], [3, 1], [3, 1], [3, 1], [1, 1], [3, 1], [3, 1], [1, 1]], [[3, 1], [4, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [1, 1]], [[2, 1], [3, 1], [3, 1], [3, 1], [3, 1], [1, 1], [2, 1], [3, 1]], [[4, 1], [2, 1], [2, 1], [3, 1], [2, 1], [3, 1], [2, 1], [1, 1]], [[2, 1], [2, 1], [1, 1], [2, 1], [4, 1], [3, 1], [3, 1], [1, 1]], [[3, 1], [3, 1], [3, 1], [3, 1], [1, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [4, 1], [2, 1], [2, 1], [3, 1], [1, 1], [2, 1], [2, 1]]]
    # repeats_10_100 = [[[4, 1], [3, 1], [3, 1], [5, 1], [4, 1], [2, 1], [3, 1], [2, 1], [2, 1], [3, 1]], [[3, 1], [4, 1], [3, 1], [3, 1], [3, 1], [3, 1], [4, 1], [2, 1], [3, 1], [2, 1]], [[1, 1], [3, 1], [3, 1], [2, 1], [5, 1], [1, 1], [3, 1], [2, 1], [4, 1], [4, 1]], [[3, 1], [3, 1], [2, 1], [1, 1], [2, 1], [3, 1], [4, 1], [4, 1], [2, 1], [2, 1]], [[1, 1], [1, 1], [1, 1], [4, 1], [2, 1], [2, 1], [1, 1], [3, 1], [1, 1], [2, 1]], [[2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [2, 1], [1, 1], [3, 1], [2, 1], [1, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [4, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[1, 1], [3, 1], [3, 1], [1, 1], [4, 1], [4, 1], [4, 1], [2, 1], [1, 1], [4, 1]], [[2, 1], [4, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1], [4, 1], [3, 1], [3, 1]], [[4, 1], [1, 1], [1, 1], [3, 1], [1, 1], [4, 1], [2, 1], [2, 1], [4, 1], [4, 1]], [[4, 1], [3, 1], [2, 1], [4, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1]], [[3, 1], [1, 1], [3, 1], [4, 1], [2, 1], [2, 1], [2, 1], [4, 1], [3, 1], [3, 1]], [[2, 1], [3, 1], [2, 1], [3, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [1, 1]], [[1, 1], [3, 1], [3, 1], [3, 1], [5, 1], [3, 1], [2, 1], [4, 1], [1, 1], [2, 1]], [[2, 1], [4, 1], [2, 1], [2, 1], [2, 1], [2, 1], [4, 1], [1, 1], [1, 1], [2, 1]], [[2, 1], [4, 1], [3, 1], [3, 1], [1, 1], [3, 1], [1, 1], [1, 1], [4, 1], [3, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [3, 1]], [[2, 1], [1, 1], [2, 1], [1, 1], [3, 1], [1, 1], [1, 1], [3, 1], [2, 1], [4, 1]], [[1, 1], [3, 1], [2, 1], [1, 1], [3, 1], [2, 1], [3, 1], [3, 1], [5, 1], [4, 1]], [[2, 1], [2, 1], [4, 1], [3, 1], [3, 1], [1, 1], [2, 1], [2, 1], [3, 1], [2, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [3, 1], [5, 1], [2, 1]], [[2, 1], [2, 1], [2, 1], [3, 1], [1, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1]], [[1, 1], [2, 1], [2, 1], [3, 1], [1, 1], [3, 1], [4, 1], [1, 1], [3, 1], [3, 1]], [[2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1], [4, 1], [4, 1], [2, 1], [1, 1]], [[3, 1], [2, 1], [3, 1], [2, 1], [3, 1], [3, 1], [2, 1], [1, 1], [1, 1], [3, 1]], [[1, 1], [2, 1], [2, 1], [2, 1], [1, 1], [3, 1], [3, 1], [3, 1], [2, 1], [3, 1]], [[3, 1], [1, 1], [4, 1], [3, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [5, 1]], [[2, 1], [4, 1], [2, 1], [4, 1], [3, 1], [3, 1], [1, 1], [4, 1], [3, 1], [4, 1]], [[2, 1], [2, 1], [5, 1], [1, 1], [2, 1], [4, 1], [3, 1], [3, 1], [2, 1], [3, 1]], [[1, 1], [4, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [1, 1], [3, 1], [2, 1]], [[2, 1], [4, 1], [1, 1], [2, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [2, 1], [5, 1], [2, 1], [3, 1]], [[3, 1], [1, 1], [3, 1], [1, 1], [1, 1], [3, 1], [1, 1], [3, 1], [2, 1], [4, 1]], [[4, 1], [2, 1], [2, 1], [1, 1], [2, 1], [2, 1], [2, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [1, 1], [4, 1], [2, 1], [2, 1], [2, 1], [1, 1], [2, 1], [2, 1], [3, 1]], [[3, 1], [3, 1], [3, 1], [3, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [1, 1]], [[3, 1], [2, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [1, 1], [2, 1], [3, 1]], [[1, 1], [3, 1], [1, 1], [1, 1], [3, 1], [1, 1], [2, 1], [1, 1], [4, 1], [3, 1]], [[3, 1], [2, 1], [2, 1], [1, 1], [2, 1], [2, 1], [3, 1], [4, 1], [4, 1], [3, 1]], [[2, 1], [1, 1], [3, 1], [3, 1], [1, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1]], [[1, 1], [3, 1], [2, 1], [2, 1], [4, 1], [3, 1], [3, 1], [3, 1], [4, 1], [4, 1]], [[3, 1], [3, 1], [1, 1], [3, 1], [2, 1], [3, 1], [3, 1], [1, 1], [2, 1], [3, 1]], [[2, 1], [1, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[2, 1], [1, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1]], [[1, 1], [3, 1], [3, 1], [3, 1], [3, 1], [5, 1], [2, 1], [1, 1], [3, 1], [2, 1]], [[2, 1], [2, 1], [2, 1], [3, 1], [2, 1], [5, 1], [2, 1], [2, 1], [1, 1], [3, 1]], [[1, 1], [2, 1], [2, 1], [2, 1], [4, 1], [3, 1], [4, 1], [2, 1], [4, 1], [3, 1]], [[3, 1], [4, 1], [2, 1], [2, 1], [3, 1], [1, 1], [4, 1], [3, 1], [2, 1], [3, 1]], [[1, 1], [2, 1], [4, 1], [4, 1], [5, 1], [2, 1], [2, 1], [2, 1], [1, 1], [
    #     2, 1]], [[3, 1], [2, 1], [4, 1], [2, 1], [2, 1], [2, 1], [1, 1], [1, 1], [1, 1], [4, 1]], [[3, 1], [4, 1], [1, 1], [1, 1], [2, 1], [3, 1], [3, 1], [4, 1], [3, 1], [4, 1]], [[2, 1], [2, 1], [1, 1], [2, 1], [3, 1], [2, 1], [3, 1], [4, 1], [1, 1], [3, 1]], [[1, 1], [3, 1], [4, 1], [3, 1], [1, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[3, 1], [1, 1], [2, 1], [4, 1], [2, 1], [2, 1], [4, 1], [1, 1], [4, 1], [2, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [2, 1], [4, 1], [1, 1], [3, 1], [1, 1], [2, 1]], [[5, 1], [2, 1], [3, 1], [1, 1], [2, 1], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1]], [[3, 1], [1, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1], [1, 1], [2, 1], [4, 1]], [[1, 1], [2, 1], [1, 1], [3, 1], [2, 1], [2, 1], [2, 1], [3, 1], [2, 1], [1, 1]], [[3, 1], [3, 1], [2, 1], [1, 1], [3, 1], [3, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[1, 1], [3, 1], [2, 1], [3, 1], [1, 1], [2, 1], [2, 1], [1, 1], [3, 1], [2, 1]], [[1, 1], [1, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [4, 1], [1, 1]], [[3, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1], [3, 1], [4, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [2, 1], [2, 1], [3, 1], [2, 1], [3, 1], [3, 1], [4, 1]], [[2, 1], [4, 1], [4, 1], [2, 1], [1, 1], [2, 1], [4, 1], [2, 1], [2, 1], [4, 1]], [[3, 1], [1, 1], [3, 1], [3, 1], [3, 1], [3, 1], [2, 1], [1, 1], [2, 1], [2, 1]], [[2, 1], [1, 1], [3, 1], [3, 1], [2, 1], [2, 1], [2, 1], [4, 1], [1, 1], [3, 1]], [[2, 1], [1, 1], [1, 1], [2, 1], [3, 1], [3, 1], [2, 1], [4, 1], [4, 1], [3, 1]], [[2, 1], [4, 1], [3, 1], [3, 1], [2, 1], [1, 1], [5, 1], [1, 1], [2, 1], [5, 1]], [[3, 1], [3, 1], [1, 1], [1, 1], [2, 1], [4, 1], [2, 1], [2, 1], [3, 1], [4, 1]], [[1, 1], [1, 1], [2, 1], [4, 1], [3, 1], [5, 1], [3, 1], [2, 1], [4, 1], [3, 1]], [[2, 1], [3, 1], [2, 1], [2, 1], [4, 1], [2, 1], [2, 1], [2, 1], [4, 1], [2, 1]], [[2, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [2, 1], [4, 1], [3, 1], [3, 1]], [[2, 1], [2, 1], [2, 1], [1, 1], [5, 1], [1, 1], [3, 1], [1, 1], [2, 1], [2, 1]], [[1, 1], [1, 1], [4, 1], [2, 1], [2, 1], [3, 1], [3, 1], [3, 1], [1, 1], [3, 1]], [[3, 1], [1, 1], [3, 1], [4, 1], [2, 1], [3, 1], [3, 1], [2, 1], [3, 1], [1, 1]], [[2, 1], [3, 1], [3, 1], [3, 1], [3, 1], [1, 1], [2, 1], [3, 1], [4, 1], [2, 1]], [[2, 1], [3, 1], [2, 1], [3, 1], [2, 1], [1, 1], [2, 1], [2, 1], [1, 1], [2, 1]], [[4, 1], [3, 1], [3, 1], [1, 1], [3, 1], [3, 1], [3, 1], [3, 1], [1, 1], [2, 1]], [[2, 1], [2, 1], [2, 1], [4, 1], [2, 1], [2, 1], [3, 1], [1, 1], [2, 1], [2, 1]], [[4, 1], [3, 1], [2, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [1, 1], [2, 1]], [[2, 1], [3, 1], [4, 1], [2, 1], [3, 1], [4, 1], [2, 1], [4, 1], [2, 1], [3, 1]], [[2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [3, 1], [3, 1], [5, 1], [4, 1], [2, 1]], [[3, 1], [3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [2, 1], [2, 1], [3, 1], [3, 1]], [[3, 1], [2, 1], [3, 1], [5, 1], [3, 1], [2, 1], [3, 1], [2, 1], [2, 1], [3, 1]], [[1, 1], [1, 1], [2, 1], [1, 1], [1, 1], [2, 1], [1, 1], [2, 1], [2, 1], [3, 1]], [[4, 1], [3, 1], [3, 1], [3, 1], [3, 1], [1, 1], [2, 1], [3, 1], [1, 1], [3, 1]], [[4, 1], [2, 1], [2, 1], [2, 1], [3, 1], [1, 1], [2, 1], [2, 1], [3, 1], [1, 1]], [[4, 1], [2, 1], [3, 1], [2, 1], [2, 1], [2, 1], [3, 1], [2, 1], [3, 1], [2, 1]], [[2, 1], [3, 1], [3, 1], [2, 1], [1, 1], [4, 1], [2, 1], [1, 1], [5, 1], [2, 1]], [[1, 1], [2, 1], [3, 1], [3, 1], [2, 1], [4, 1], [2, 1], [1, 1], [3, 1], [4, 1]], [[2, 1], [1, 1], [4, 1], [1, 1], [1, 1], [3, 1], [2, 1], [2, 1], [3, 1], [2, 1]], [[1, 1], [4, 1], [4, 1], [3, 1], [3, 1], [5, 1], [2, 1], [2, 1], [3, 1], [4, 1]], [[4, 1], [2, 1], [3, 1], [1, 1], [2, 1], [2, 1], [1, 1], [4, 1], [2, 1], [2, 1]], [[3, 1], [4, 1], [3, 1], [5, 1], [2, 1], [3, 1], [3, 1], [4, 1], [3, 1], [2, 1]], [[2, 1], [3, 1], [4, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [5, 1], [1, 1]], [[2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [2, 1], [3, 1], [4, 1], [3, 1], [3, 1]], [[2, 1], [3, 1], [3, 1], [4, 1], [5, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1]], [[3, 1], [2, 1], [2, 1], [4, 1], [2, 1], [3, 1], [2, 1], [3, 1], [4, 1], [2, 1]], [[3, 1], [3, 1], [2, 1], [3, 1], [3, 1], [3, 1], [2, 1], [3, 1], [1, 1], [2, 1]]]

    # repeats = repeats_5_100

    # def make_env(repeat, env_fn):
    #     # Create the environment
    #     num_machines = 3
    #     num_jobs = 5
    #     max_repeats = 5
    #     cost_list = [5, 1, 2, 10]
    #     profit_per_time = 10
    #     max_time = 50

    #     return env_fn(
    #         machine_config_path=f"instances/Machines/v0-{str(num_machines)}x{str(num_jobs)}.json",
    #         job_config_path=f"instances/Jobs/v0-{str(num_machines)}x{str(num_jobs)}.json",
    #         job_repeats_params=repeat,
    #         render_mode="seaborn",
    #         cost_deadline_per_time=cost_list[0],
    #         cost_hole_per_time=cost_list[1],
    #         cost_processing_per_time=cost_list[2],
    #         cost_makespan_per_time=cost_list[3],
    #         profit_per_time=profit_per_time,
    #         target_time=None,
    #         test_mode=True,
    #         max_time=max_time,
    #         num_of_types=4
    #     )

    # pdr_envs = [make_env(repeat, PDRenv) for repeat in repeats]
    # rl_envs = [make_env(repeat, RLenv) for repeat in repeats]

    # # Evaluate the PDR
    # pdr = ETD
    # results = eval_pdr(pdr, pdr_envs, render=True, verbose=True)

    # Compare the PDRs
    pdrs = [MWKR, CR, FDD_over_MWKR, LWKR_MOD, LWKR_SPT, ETD]
    log_dir = "./experiments/tmp/0-paper-8x12-ppo"
    model_path = './logs/tmp/0-paper-3x5-1m-lr5/best_model.zip'

    # policy_kwargs = dict(
    #     # activation_fn=th.nn.LeakyReLU,
    #     net_arch=dict(
    #         pi=[256, 128, 64],
    #         vf=[256, 128, 64]
    #     )
    # )
    # model = MaskablePPO.load(model_path, policy_kwargs=policy_kwargs)

    # compare_pdr_model(pdrs, pdr_envs, model, rl_envs, deterministic=False,
    #                   sample_times=100, log_dir=log_dir, verbose=False, render=False)

    objs = [
        # 'makespan',
        # 'total_tardiness',
        # 'processing_time',
        # 'idle_time',
        'makespan_cost',
        'tard_cost',
        'idle_time_cost',
        'total_cost'
    ]
    # plot_pdr_comparison(pdrs, objs, with_model=True, log_dir=log_dir, pie=True, save_img=True)


    # 4개의  cost 결과를 하나의 plot으로 만듦
    plot_combined_comparison(pdrs, objs, with_model=True, log_dir=log_dir, pie=True, save_img=True)
