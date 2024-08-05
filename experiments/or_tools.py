import collections
from ortools.sat.python import cp_model
from scheduler_env.customEnv_repeat import SchedulingEnv
import matplotlib.pyplot as plt
import numpy as np
import time
import copy


def solve_with_ortools(env,
                       objective='makespan',
                       cost_weights={
                           'tard': 5,
                           'idle_time': 1,
                           'makespan': 10
                       },
                       SCALE=10000, time_limit=60.0,
                       n_workers=8,
                       copy_env=True,
                       verbose=False
                       ):  # Scaling factor for float approximation
    if verbose:
        print(f'Objective: {objective}')
    model = cp_model.CpModel()
    if copy_env:
        env = copy.deepcopy(env)

    jobs = env.custom_scheduler.jobs
    machines = env.custom_scheduler.machines

    # Constants
    max_time = sum(op.duration for job_list in jobs for job in job_list for op in job.operation_queue)

    # Variables
    task_starts = {}
    task_ends = {}
    task_machines = {}
    # for no overlap constraint
    machine_to_intervals = collections.defaultdict(list)
    # for deadline compliance
    deadline_met = {}
    # for MSE objective
    job_earliness = {}
    job_scaled_earliness = {}
    # for cost objective
    tardiness = {}
    idle_time = {}
    machine_starts = {}
    machine_ends = {}
    total_operation_time = model.NewIntVar(0, max_time, 'total_operation_time')
    model.Add(total_operation_time == max_time)

    for m in range(len(machines)):
        machine_starts[m] = model.NewIntVar(0, max_time, f'machine_{m}_start')
        machine_ends[m] = model.NewIntVar(0, max_time, f'machine_{m}_end')

    n_operations = 0
    for job_list in jobs:
        for job in job_list:
            for i, op in enumerate(job.operation_queue):
                n_operations += 1
                task_starts[(job.name, job.index, op.index)] = model.NewIntVar(
                    0, max_time, f'start_{job.name}-{job.index}_op{op.index}')
                task_ends[(job.name, job.index, op.index)] = model.NewIntVar(
                    0, max_time, f'end_{job.name}-{job.index}_op{op.index}')
                task_machines[(job.name, job.index, op.index)] = model.NewIntVar(
                    0, len(machines) - 1, f'machine_{job.name}-{job.index}_op{op.index}')

                deadline_met[(job.name, job.index, op.index)] = model.NewBoolVar(
                    f'deadline_met_end_{job.name}-{job.index}_op{op.index}')

                # Add duration constraint
                model.Add(task_ends[(job.name, job.index, op.index)] ==
                          task_starts[(job.name, job.index, op.index)] + op.duration)

                # Machine capability constraint
                capable_machines = [m for m in range(len(machines)) if machines[m].can_process_operation(op.type)]
                if capable_machines:
                    model.AddAllowedAssignments([task_machines[(job.name, job.index, op.index)]], [(m,)
                                                for m in capable_machines])
                else:
                    print(
                        f"Warning: No capable machine found for {job.name}-{job.index}, operation {op.index}, type {op.type}")
                    return None, None  # If no machine can process this operation, the problem is infeasible

                if i > 0:
                    # Precedence constraints
                    model.Add(task_starts[(job.name, job.index, op.index)] >=
                              task_ends[(job.name, job.index, op.index - 1)])

                # Machine constraints
                # Create interval variable for each machine this task could be assigned to
                for m in capable_machines:
                    is_present = model.NewBoolVar(f'is_on_machine_{m}_{job.name}-{job.index}_op{op.index}')
                    interval = model.NewOptionalIntervalVar(
                        task_starts[(job.name, job.index, op.index)],
                        op.duration,
                        task_ends[(job.name, job.index, op.index)],
                        is_present,
                        f'interval_{job.name}-{job.index}_op{op.index}_machine{m}'
                    )
                    machine_to_intervals[m].append(interval)

                    # Link the interval to the machine assignment
                    model.Add(task_machines[(job.name, job.index, op.index)] == m).OnlyEnforceIf(is_present)
                    model.Add(task_machines[(job.name, job.index, op.index)] != m).OnlyEnforceIf(is_present.Not())

                    # Update machine start and end times only if the operation is on this machine
                    start_time = task_starts[(job.name, job.index, op.index)]
                    end_time = task_ends[(job.name, job.index, op.index)]
                    model.Add(machine_starts[m] <= start_time).OnlyEnforceIf(is_present)
                    model.Add(machine_ends[m] >= end_time).OnlyEnforceIf(is_present)
            # MSE calculations
            job_earliness[(job.name, job.index)] = model.NewIntVar(-max_time * SCALE,
                                                                   max_time * SCALE, f'earliness_{job.name}-{job.index}')

            last_op = job.operation_queue[-1]

            job_completion_time = model.NewIntVar(0, max_time, f'job_{job.name}_{job.index}_completion_time')
            model.Add(job_completion_time == task_ends[(job.name, job.index, last_op.index)])

            # Calculate earliness (can be negative)
            model.Add(job_earliness[(job.name, job.index)] ==
                      job.deadline - job_completion_time)

            # For now, we'll set scaled_earliness equal to earliness (no scaling yet)
            job_scaled_earliness[(job.name, job.index)] = model.NewIntVar(-max_time * SCALE,
                                                                          max_time * SCALE, f'scaledearliness{job.name}-{job.index}')

            # Calculate scaled earliness
            total_job_length = sum(op.duration for op in job.operation_queue)
            scale_factor = round((total_job_length * SCALE) / job.deadline)  # Integer division
            model.AddMultiplicationEquality(
                job_scaled_earliness[(job.name, job.index)],
                job_earliness[(job.name, job.index)],
                scale_factor
            )

            # Calculate tardiness as the maximum lateness of any operation in the job
            tardiness[(job.name, job.index)] = model.NewIntVar(0, max_time, f'tardiness_{job.name}-{job.index}')
            model.Add(tardiness[(job.name, job.index)] >= job_completion_time - job.deadline)
            model.Add(tardiness[(job.name, job.index)] >= 0)

    # No overlap constraint
    for machine in machine_to_intervals:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Objective: Minimize makespan
    makespan = model.NewIntVar(0, max_time, 'makespan')
    model.AddMaxEquality(makespan, [task_ends[(job.name, job.index, op.index)]
                                    for job_list in jobs for job in job_list for op in job.operation_queue])

    # Objective : Deadline Compliance
    deadline_compliance = model.NewIntVar(0, sum(len(job_list) for job_list in jobs), 'deadline_compliance')
    deadline_met_list = []

    # constraints and counting
    for job_list in jobs:
        for job in job_list:
            job_meets_deadline = model.NewBoolVar(f'job_{job.name}_{job.index}_meets_deadline')
            operation_meets_deadline_list = []

            for op in job.operation_queue:
                # Assume each operation inherits the job's deadline
                # You might need to adjust this if operations have individual deadlines
                model.Add(task_ends[(job.name, job.index, op.index)] <= job.deadline).OnlyEnforceIf(
                    deadline_met[(job.name, job.index, op.index)])
                model.Add(task_ends[(job.name, job.index, op.index)] > job.deadline).OnlyEnforceIf(
                    deadline_met[(job.name, job.index, op.index)].Not())
                operation_meets_deadline_list.append(deadline_met[(job.name, job.index, op.index)])

            # A job meets the deadline if all its operations meet the deadline
            model.AddMinEquality(job_meets_deadline, operation_meets_deadline_list)
            deadline_met_list.append(job_meets_deadline)

    model.Add(deadline_compliance == sum(deadline_met_list))

    # Objective: Mean Scaled Earliness
    # Calculate Mean Scaled Earliness
    total_scaled_earliness = model.NewIntVar(-max_time * len(jobs) * SCALE,
                                             max_time * len(jobs) * SCALE, 'total_scaled_earliness')
    model.Add(total_scaled_earliness == sum(job_scaled_earliness.values()))

    mse = model.NewIntVar(-max_time * SCALE, max_time * SCALE, 'mse')
    model.AddDivisionEquality(mse, total_scaled_earliness, len(jobs))

    # Objective: Cost
    # Calculate tardiness
    total_tardiness = model.NewIntVar(0, max_time * len(jobs), 'total_tardiness')
    model.Add(total_tardiness == sum(tardiness.values()))

    # Calculate total idle time
    total_machine_time = model.NewIntVar(0, max_time * len(machines), 'total_machine_time')
    machine_times = []

    for m in range(len(machines)):
        machine_time = model.NewIntVar(0, max_time, f'machine_{m}_time')
        model.Add(machine_time == machine_ends[m] - machine_starts[m])
        machine_times.append(machine_time)

    model.Add(total_machine_time == sum(machine_times))

    total_idle_time = model.NewIntVar(0, max_time * len(machines), 'total_idle_time')
    model.Add(total_idle_time == total_machine_time - total_operation_time)

    # Set objective
    if objective == 'makespan':
        model.Minimize(makespan)
    elif objective == 'deadline':
        model.Maximize(deadline_compliance)
    elif objective == 'MSE':
        model.Maximize(mse)
    elif objective == 'cost':
        model.Minimize(cost_weights['tard'] * total_tardiness
                       + cost_weights['idle_time'] * total_idle_time
                       + cost_weights['makespan'] * makespan)
    else:
        raise ValueError(f"Objective {objective} not supported")

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit  # Set a time limit of 60 seconds
    solver.parameters.num_search_workers = n_workers  # Adjust based on your CPU cores
    # solver.parameters.log_search_progress = True

    # Start timing
    start_time = time.time()
    status = solver.Solve(model)
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if verbose:
            print(f'Time elapsed: {elapsed_time} | Status: {solver.StatusName(status)}')
            
        for m in range(len(machines)):
            print(f'Machine {m} start: {solver.Value(machine_starts[m])}, end: {solver.Value(machine_ends[m])}')
            
        print(f'Total operation time: {solver.Value(total_operation_time)}')
        print(f'Total machine time: {solver.Value(total_machine_time)}')
        print(f'Total idle time: {solver.Value(total_idle_time)}')
        # Extract solution
        solution = []
        i = 0
        env.custom_scheduler.last_finish_time = 0
        for job_list in jobs:
            for job in job_list:
                for op in job.operation_queue:
                    solution.append({
                        'job': job.index,
                        'operation': op.index,
                        'start': solver.Value(task_starts[(job.name, job.index, op.index)]),
                        'end': solver.Value(task_ends[(job.name, job.index, op.index)]),
                        'machine': solver.Value(task_machines[(job.name, job.index, op.index)])
                    })
                    op.sequence = i
                    op.start = solver.Value(task_starts[(job.name, job.index, op.index)])
                    op.finish = solver.Value(task_ends[(job.name, job.index, op.index)])
                    op.machine = solver.Value(task_machines[(job.name, job.index, op.index)])

                    i += 1

                    env.custom_scheduler.last_finish_time = max(
                        env.custom_scheduler.last_finish_time, solver.Value(task_ends[(job.name, job.index, op.index)]))

                    env.custom_scheduler.current_schedule.append(op)
        tard_cost = solver.Value(total_tardiness) * cost_weights['tard']
        idle_time_cost = solver.Value(total_idle_time) * cost_weights['idle_time']
        makespan_cost = solver.Value(makespan) * cost_weights['makespan']
        objs = {
            'makespan': solver.Value(makespan),
            'deadline_compliance': solver.Value(deadline_compliance),
            'MSE': solver.Value(mse) / SCALE,
            'tard_cost': tard_cost,
            'idle_time_cost': idle_time_cost,
            'makespan_cost': makespan_cost,
            'total_cost': tard_cost + idle_time_cost + makespan_cost
        }
        return solution, objs, solver.StatusName(status), elapsed_time
    else:
        return None, None, solver.StatusName(status), elapsed_time


if __name__ == '__main__':
    cost_weights = {
        'tard': 5,
        'idle_time': 1,
        'makespan': 10
    }
    # Create environment
    env = SchedulingEnv(machine_config_path="instances/Machines/v0-8.json", job_config_path="instances/Jobs/v0-12-repeat-hard.json",
                        job_repeats_params=[(8, 1)] * 12, test_mode=True)
    env.reset()
    objective='cost'
    copy_env = False
    solution, objs, status, elapsed_time = solve_with_ortools(
        env, objective=objective, cost_weights=cost_weights, copy_env=copy_env, time_limit=600.0)
    print(f"Status: {status}, Elapsed time: {elapsed_time}")
    if solution:
        print(f"Objective values: {objs}")
        if not copy_env:
            info = env._get_info()
            n_jobs = 0
            earliness = [job.deadline -
                         job.operation_queue[-1].finish for job_list in env.custom_scheduler.jobs for job in job_list]
            tardiness = [max(0, -earliness[i]) for i in range(len(earliness))]
            total_tardiness = sum(tardiness)
            if objective != 'cost':
                total_op_time = sum(op.duration for op in env.custom_scheduler.current_schedule)
                # machine time = finish - start
                total_machine_time = 0
                for m in range(len(env.custom_scheduler.machines)):
                    machine_operations = [op for op in env.custom_scheduler.current_schedule if op.machine == m]
                    machine_start = min(op.start for op in machine_operations)
                    machine_end = max(op.finish for op in machine_operations)
                    total_machine_time += machine_end - machine_start
                print('total_machine_time', total_machine_time)
                total_idle_time = total_machine_time - total_op_time
                print('total_idle_time', total_idle_time)
                idle_time_cost = total_idle_time * cost_weights['idle_time']
                print('Real cost values:')
                tard_cost = total_tardiness * cost_weights['tard']
                makespan_cost = objs['makespan'] * cost_weights['makespan']
                total_cost = tard_cost + idle_time_cost + makespan_cost
                print(f'\ttard_cost {tard_cost} | idle_time_cost {idle_time_cost} | makespan_cost {makespan_cost} | total_cost {total_cost}')

            print('job_deadline', info['job_deadline'])
            print(f'job_tardiness {tardiness} | total {total_tardiness}')
            print(f'job_earliness {earliness}')
            print(f'current_repeats {info["current_repeats"]}, n_jobs {n_jobs}')
            env.render()

    else:
        print("No solution found")
