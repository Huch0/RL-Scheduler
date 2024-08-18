import numpy as np
from scheduler_env.customEnv_repeat import SchedulingEnv
from scheduler_env.customEnv_No_heatmap import SchedulingEnvNoHeatmap

def make_env(
    num_machines,
    num_jobs,
    max_repeats,
    repeat_means,
    repeat_stds,
    test_mode=False,
    cost_list=None,
    profit_per_time=10,
    max_time=150,
    has_heatmap=True
):
    if cost_list is None:
        cost_list = [5, 1, 2, 10]  # Default costs: [cost_deadline_per_time, cost_hole_per_time, cost_processing_per_time, cost_makespan_per_time]

    assert len(cost_list) == 4, "cost_list must contain exactly 4 elements."

    # 3. job_repeats_params 생성
    job_repeats_params = [(repeat_means[i], repeat_stds[i]) for i in range(len(repeat_means))]

    # 4. 환경 생성
    env = SchedulingEnv(
        machine_config_path= "instances/Machines/v0-" + str(num_machines) + ".json",  
        job_config_path= "instances/Jobs/v0-" + str(num_jobs) + "x" + str(max_repeats) + ".json", 
        job_repeats_params=job_repeats_params,
        render_mode="seaborn",
        cost_deadline_per_time=cost_list[0],
        cost_hole_per_time=cost_list[1],
        cost_processing_per_time=cost_list[2],
        cost_makespan_per_time=cost_list[3],
        profit_per_time=profit_per_time,
        target_time=None,
        test_mode=test_mode,
        max_time=max_time
    )

    if not has_heatmap:
        env = SchedulingEnvNoHeatmap(
            machine_config_path= "instances/Machines/v0-" + str(num_machines) + ".json",  
            job_config_path= "instances/Jobs/v0-" + str(num_jobs) + "x" + str(max_repeats) + ".json", 
            job_repeats_params=job_repeats_params,
            render_mode="seaborn",
            cost_deadline_per_time=cost_list[0],
            cost_hole_per_time=cost_list[1],
            cost_processing_per_time=cost_list[2],
            cost_makespan_per_time=cost_list[3],
            profit_per_time=profit_per_time,
            target_time=None,
            test_mode=test_mode,
            max_time=max_time
        )
    # env 이름 생성
    env_name = f"Env_{str(num_machines)}_{str(num_jobs)}_{int(np.mean(repeat_means))}_{int(np.mean(repeat_stds))}_p{str(profit_per_time)}"

    return env, env_name

if __name__ == "__main__":
    # Test the function
    env, env_name = make_env(
        num_machines=5,
        num_jobs_with_repeats=15,
        repeat_means=[3] * 5,
        repeat_stds=[1] * 5,
        test_mode=False,
        cost_list=[5, 1, 2, 10],
        profit_per_time=10
    )

    print(f"Environment Name: {env_name}")
    # Add additional checks or functionality as needed