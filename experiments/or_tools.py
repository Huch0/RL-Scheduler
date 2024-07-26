import collections
from ortools.sat.python import cp_model
from scheduler_env.customEnv_repeat import SchedulingEnv
import matplotlib.pyplot as plt
import numpy as np
import time

def solve_with_ortools(env, objective='makespan', SCALE=10000, time_limit=60.0):  # Scaling factor for float approximation
    print(f'Objective: {objective}')
    model = cp_model.CpModel()

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

            # MSE calculations
            job_earliness[(job.name, job.index)] = model.NewIntVar(-max_time * SCALE,
                                                                   max_time * SCALE, f'earliness_{job.name}-{job.index}')

            # Calculate earliness (can be negative)
            last_op = job.operation_queue[-1]
            model.Add(job_earliness[(job.name, job.index)] ==
                      job.deadline - task_ends[(job.name, job.index, last_op.index)])

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
    # No overlap constraint
    for machine in machine_to_intervals:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Objective: Minimize makespan
    makespan = model.NewIntVar(0, max_time, 'makespan')
    model.AddMaxEquality(makespan, [task_ends[(job.name, job.index, op.index)]
                         for job_list in jobs for job in job_list for op in job.operation_queue])

    # Objective : Deadline Compliance
    deadline_compliance = model.NewIntVar(0, sum(len(job.operation_queue)
                                          for job_list in jobs for job in job_list), 'deadline_compliance')
    deadline_met_list = []
    # constraints and counting
    for job_list in jobs:
        for job in job_list:
            for op in job.operation_queue:
                # Assume each operation inherits the job's deadline
                # You might need to adjust this if operations have individual deadlines
                model.Add(task_ends[(job.name, job.index, op.index)] <= job.deadline).OnlyEnforceIf(
                    deadline_met[(job.name, job.index, op.index)])
                model.Add(task_ends[(job.name, job.index, op.index)] > job.deadline).OnlyEnforceIf(
                    deadline_met[(job.name, job.index, op.index)].Not())
                deadline_met_list.append(deadline_met[(job.name, job.index, op.index)])

    model.Add(deadline_compliance == sum(deadline_met_list))

    # Objective: Mean Scaled Earliness
    # Calculate Mean Scaled Earliness
    total_scaled_earliness = model.NewIntVar(-max_time * len(jobs) * SCALE,
                                             max_time * len(jobs) * SCALE, 'total_scaled_earliness')
    model.Add(total_scaled_earliness == sum(job_scaled_earliness.values()))

    mse = model.NewIntVar(-max_time * SCALE, max_time * SCALE, 'mse')
    model.AddDivisionEquality(mse, total_scaled_earliness, len(jobs))

    # Set objective
    if objective == 'makespan':
        model.Minimize(makespan)
    elif objective == 'deadline':
        model.Maximize(deadline_compliance)
    elif objective == 'MSE':
        model.Maximize(mse)
    else:
        raise ValueError(f"Objective {objective} not supported")

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit  # Set a time limit of 60 seconds
    solver.parameters.num_search_workers = 8  # Adjust based on your CPU cores
    # solver.parameters.log_search_progress = True
    
    # Start timing
    start_time = time.time()
    status = solver.Solve(model)
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'Time elapsed: {elapsed_time} | Status: {solver.StatusName(status)}')
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
        # print(solution)
        objs = [solver.Value(makespan), solver.Value(deadline_compliance), solver.Value(mse) / SCALE]
        return solution, objs
    else:
        return None, None


if __name__ == '__main__':
    # Create environment
    env = SchedulingEnv(machine_config_path="instances/Machines/v0-8.json", job_config_path="instances/Jobs/v0-12-repeat.json",
                        job_repeats_params=[(3, 1)] * 12, weight_final_time=0, weight_job_deadline=0.01, weight_op_rate=0, test_mode=False)
    env.reset()

    solution, objs = solve_with_ortools(env, objective='makespan')
    if solution:
        earliness = [job.deadline - job.operation_queue[-1].finish for job_list in env.custom_scheduler.jobs for job in job_list]
        scaled_earliness = []
        for job_list in env.custom_scheduler.jobs:
            for job in job_list:
                total_duration = sum([op.duration for op in job.operation_queue])
                deadline = job.deadline
                job_earliness = job.deadline - job.operation_queue[-1].finish
                scaling_factor = total_duration / deadline
                scaled_value = scaling_factor * job_earliness
                scaled_earliness.append(scaled_value)

                # Print detailed debug information
                # print(f"Job Total Duration: {total_duration}, Job Deadline: {deadline}, Job Earliness: {job_earliness}, Scaling Factor: {scaling_factor}, Scaled Earliness: {scaled_value}")
        info = env._get_info()
        print(f"makespan: {objs[0]}, deadline compliance: {objs[1]}, MSE: {objs[2]}")
        print('job_deadline', info['job_deadline'])
        print(f'job_earliness {earliness}')
        print(f'job_scaled_earliness {scaled_earliness}')
        print(f'MSE: {np.mean(scaled_earliness)}')
        print('current_repeats', info['current_repeats'])

        env.render()
    else:
        print("No solution found")
