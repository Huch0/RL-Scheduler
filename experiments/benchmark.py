from experiments.or_tools import solve_with_ortools
from utils.tb_logger import CustomLogger
import time
import csv


def benchmark(env, rl_model, iteration=100, or_kwargs=dict(), log_dir='./experiments/tmp/1', verbose=True):
    writer = CustomLogger(log_dir)

    for i in range(iteration):
        if verbose:
            print(f'Iteration {i+1}/{iteration}')
        obs, info = env.reset()
        n_jobs = sum(info['current_repeats'])

        # Solve with OR-Tools (makespan)
        _, ms_objs, ms_status, ms_time = solve_with_ortools(env, objective='makespan', **or_kwargs)
        # Solve with OR-Tools (deadline compliance)
        _, dc_objs, dc_status, dc_time = solve_with_ortools(env, objective='deadline', **or_kwargs)

        # Solve with RL model
        done = False
        start_time = time.time()
        while not done:
            action_masks = env.action_masks()
            action, _states = rl_model.predict(obs, action_masks=action_masks, deterministic=False)
            obs, reward, te, tr, info = env.step(action)
            done = te or tr
        end_time = time.time()
        rl_time = end_time - start_time
        if tr:
            print("Goal reached!", "final score=", reward)
            print('job_deadline', info['job_deadline'])
            print('job_time_exceeded', info['job_time_exceeded'])
            print('current_repeats', info['current_repeats'])
            raise ValueError('Truncated')

        rl_ms = info['finish_time']
        rl_dc = sum([1 if e >= 0 else 0 for e in info['job_earliness']])

        rl_mse = env._calculate_final_reward()  # Modify later

        # dc to ratio
        ms_objs['deadline_compliance'] = float(ms_objs['deadline_compliance']) / n_jobs
        dc_objs['deadline_compliance'] = float(dc_objs['deadline_compliance']) / n_jobs
        rl_dc /= n_jobs

        repeats = info['current_repeats']

        # Log results to TensorBoard
        writer.add_text('Repeats', str(repeats), i)
        writer.add_scalar('Makespan/OR-ms', ms_objs['makespan'], i)
        writer.add_scalar('DeadlineCompliance/OR-ms', ms_objs['deadline_compliance'], i)
        writer.add_scalar('MSE/OR-ms', ms_objs['MSE'], i)
        writer.add_text('Status/OR-ms', ms_status, i)
        writer.add_scalar('Time/OR-ms', ms_time, i)

        writer.add_scalar('Makespan/OR-dc', dc_objs['makespan'], i)
        writer.add_scalar('DeadlineCompliance/OR-dc', dc_objs['deadline_compliance'], i)
        writer.add_scalar('MSE/OR-dc', dc_objs['MSE'], i)
        writer.add_text('Status/OR-dc', dc_status, i)
        writer.add_scalar('Time/OR-dc', dc_time, i)

        writer.add_scalar('Makespan/RL', rl_ms, i)
        writer.add_scalar('DeadlineCompliance/RL', rl_dc, i)
        writer.add_scalar('MSE/RL', rl_mse, i)
        writer.add_scalar('Time/RL', rl_time, i)

        if verbose:
            writer.print_summary(i)

    writer.close()


if __name__ == '__main__':
    from scheduler_env.customEnv_repeat import SchedulingEnv
    from sb3_contrib import MaskablePPO
    import numpy as np

    # Create environment
    env = SchedulingEnv(machine_config_path="instances/Machines/v0-8.json", job_config_path="instances/Jobs/v0-12-repeat.json",
                        job_repeats_params=[(3, 1)] * 12, weight_final_time=0, weight_job_deadline=0.01, weight_op_rate=0, test_mode=False)
    env.reset()

    # Load RL model
    model = MaskablePPO.load('logs/tmp/1/best_model.zip')

    # Set OR-Tools kwargs
    or_kwargs = {
        'SCALE': 10000,
        'time_limit': 10.0,
        'n_workers': 8,
        'copy_env': True,
        'verbose': False
    }

    # Benchmark
    benchmark(env, model, iteration=10, or_kwargs=or_kwargs, log_dir='./experiments/tmp/1')
