import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.tb_logger import CustomLogger
from scheduler_env.pdr_env import SchedulingEnv


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
    Flow Due Data / Most Work Remaining
    """
    pass


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


def compare_pdrs(pdrs, evns, log_dir='./experiments/tmp/1', verbose=False):
    """
    Compare the given PDRs on the given environments.
    """
    writer = CustomLogger(log_dir=log_dir)

    if verbose:
        print(f'Comparing PDRs: {[pdr.__name__ for pdr in pdrs]}')

    # Log the results for each PDR
    for pdr in pdrs:
        results = eval_pdr(pdr, envs, render=False, verbose=verbose)

        for key, values in results.items():
            for i, value in enumerate(values):
                writer.add_scalar(f'{pdr.__name__}/{key}', value, i)

    writer.close()
    return


def plot_pdr_comparison(pdrs, log_dir='./experiments/tmp/1'):
    # Read the CSV file
    csv_file_path = os.path.join(log_dir, 'results.csv')
    df = pd.read_csv(csv_file_path)

    # Set up the 8 subplots
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    objs = ['makespan', 'total_tardiness', 'processing_time', 'idle_time',
            'makespan_cost', 'tard_cost', 'idle_time_cost', 'total_cost']

    for i, obj in enumerate(objs):
        ax = axs[i // 4, i % 4]
        data = []
        for pdr in pdrs:
            column_name = f'{pdr.__name__}/{obj}'
            pdr_data = df[column_name]
            pdr_df = pd.DataFrame({obj: pdr_data, 'PDR': pdr.__name__})
            data.append(pdr_df)

        combined_data = pd.concat(data)
        sns.boxplot(x='PDR', y=obj, data=combined_data, ax=ax)
        ax.set_title(obj)
        ax.set_xlabel('PDRs')
        ax.set_ylabel(obj.replace('_', ' ').title())

    plt.tight_layout()
    plt.show()


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


if __name__ == "__main__":
    # # Random seed
    # np.random.seed(0)

    # n_jobs = 12
    # eval_instances = 10
    # mean, std = 3, 1

    # # Get 12 integers from Gaussian Distribution / mean : 3 std : 1
    # repeats = [[[max(1, int(np.random.normal(mean, std)))] for _ in range(12)] for _ in range(eval_instances)]
    # print(repeats)

    repeats = [[[3] for _ in range(12)],
               [[4], [3], [3], [5], [4], [2], [3], [2], [2], [3], [3], [4]], [[3], [3], [3], [3], [4], [2], [3], [2], [1], [3], [3], [2]], [[5], [1], [3], [2], [4], [4], [3], [3], [2], [1], [2], [3]], [[4], [4], [2], [2], [1], [1], [1], [4], [2], [2], [1], [3]], [[1], [2], [2], [3], [2], [1], [2], [3], [3], [3], [2], [
                   2]], [[2], [2], [2], [1], [3], [2], [1], [3], [2], [3], [3], [3]], [[4], [1], [3], [2], [2], [2], [2], [3], [1], [3], [3], [1]], [[4], [4], [4], [2], [1], [4], [2], [4], [3], [3], [3], [3]], [[3], [4], [3], [3], [4], [1], [1], [3], [1], [4], [2], [2]], [[4], [4], [4], [3], [2], [4], [2], [3], [3], [2], [3], [3]]]

    def make_env(repeat):
        # Create the environment
        num_machines = 8
        num_jobs = 12
        max_repeats = 12
        cost_list = [5, 1, 2, 10]
        profit_per_time = 10
        max_time = 150

        return SchedulingEnv(
            machine_config_path="instances/Machines/v0-" + str(num_machines) + ".json",
            job_config_path="instances/Jobs/v0-" + str(num_jobs) + "x" + str(max_repeats) + ".json",
            job_repeats_params=repeat,
            render_mode="seaborn",
            cost_deadline_per_time=cost_list[0],
            cost_hole_per_time=cost_list[1],
            cost_processing_per_time=cost_list[2],
            cost_makespan_per_time=cost_list[3],
            profit_per_time=profit_per_time,
            target_time=None,
            test_mode=True,
            max_time=max_time
        )

    envs = [make_env(repeat) for repeat in repeats]

    # # Evaluate the PDR
    pdr = LWKR_MOD
    # results = eval_pdr(pdr, envs, render=True, verbose=True)

    # Compare the PDRs
    pdrs = [MWKR, CR, LWKR_MOD, LWKR_SPT]
    log_dir = './experiments/tmp/1'
    # compare_pdrs(pdrs, envs, log_dir=log_dir, verbose=True)

    plot_pdr_comparison(pdrs, log_dir=log_dir)
