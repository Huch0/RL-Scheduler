# test_etd_calculate.py

import numpy as np
import pytest

from rl_scheduler.RJSPEnv.Env import RJSPEnv
from rl_scheduler.RJSPEnv.Scheduler import customRepeatableScheduler


@pytest.fixture
def setup_env():
    # Paths to the configuration files (use small instances for testing)
    machine_config_path = "rl_scheduler/instances/Machines/v0-test.json"
    job_config_path = "rl_scheduler/instances/Jobs/v0-test.json"

    # Job repeats parameters (mean, std) for each job
    job_repeats_params = [(2, 0)] * 3  # 2 jobs with 1 repeat each for simplicity

    # Create the environment
    env = RJSPEnv(
        machine_config_path=machine_config_path,
        job_config_path=job_config_path,
        job_repeats_params=job_repeats_params,
        max_time=50,  # Adjust max_time as needed
        test_mode=True,  # Set test_mode to True for consistent behavior
        sample_mode="test",  # Use 'test' sample mode for repeatability
    )

    # Reset the environment to get the initial observation
    observation, info = env.reset()
    return env, observation, info


def test_etd_calculation(setup_env):
    env, observation, info = setup_env
    scheduler = env.custom_scheduler

    # Initial ETD values before any action
    initial_etd = [job_list[0].estimated_tardiness for job_list in scheduler.jobs]
    print("Initial ETD:", initial_etd)

    # Expected initial ETD values
    # 아래에서 Job이 끝났을 때 첫 번째 Job의 Tardiness로 되는 것은 개선의 여지가 있음
    expected_initial_etd = [
        [-166.66666666666666, -133.33333333333331, -100.0],
        [-300.0, -100.0, -83.33333333333333],
        [-283.3333333333333, -33.33333333333333, -50.0],
        [-600.0, -33.33333333333333, -33.33333333333333],
        [-600.0, -33.33333333333333, 0.0],
        [-500.0, 100.0, 66.66666666666666],
        [-300.0, 200.0, 200.0],
        [-300.0, 250.0, 400.0],
        [-300.0, -100.0, 500.0],
        [-300.0, -100.0, -283.3333333333333],
        [-300.0, -100.0, -333.3333333333333],
        [-300.0, -66.66666666666666, -400.0],
        [-300.0, 0.0, -350.0],
        [-300.0, 400.0, -250.0],
    ]

    assert initial_etd == expected_initial_etd[0], "Initial ETD values are incorrect."

    # Take an action and check ETD again
    # Our test case trejectory :
    # 1. Schedule operation 1_1_1 on machine 2
    # 2. Schedule operation 1_1_2 on machine 2
    # 3. Schedule operation 1_2_1 on machine 1
    # 4. Schedule operation 3_1_1 on machine 1
    # 5. Schedule operation 2_1_1 on machine 2
    # 6. Schedule operation 1_2_2 on machine 2
    # 7. Schedule operation 3_1_2 on machine 2
    # 8. Schedule operation 2_1_2 on machine 2
    # 9. Schedule operation 3_1_3 on machine 1
    # 10. Schedule operation 3_2_1 on machine 1
    # 11. Schedule operation 3_2_2 on machine 2
    # 12. Schedule operation 2_2_1 on machine 2
    # 13. Schedule operation 2_2_2 on machine 2
    # 14. Schedule operation 3_2_3 on machine 1

    actions = [
        np.int64(3),  # 1
        np.int64(3),  # 2
        np.int64(0),  # 3
        np.int64(2),  # 4
        np.int64(4),  # 5
        np.int64(3),  # 6
        np.int64(5),  # 7
        np.int64(4),  # 8
        np.int64(2),  # 9
        np.int64(2),  # 10
        np.int64(5),  # 11
        np.int64(4),  # 12
        np.int64(4),  # 13
        np.int64(2),  # 14, finished
    ]

    expected_schedule_buffer = [
        [[0, 0], [0, 0], [0, 0]],
        [[0, 1], [0, 0], [0, 0]],
        [[1, 0], [0, 0], [0, 0]],
        [[1, 1], [0, 0], [0, 0]],
        [[1, 1], [0, 0], [0, 1]],
        [[1, 1], [0, 1], [0, 1]],
        [[-1, -1], [0, 1], [0, 1]],
        [[-1, -1], [0, 1], [0, 2]],
        [[-1, -1], [1, 0], [0, 2]],
        [[-1, -1], [1, 0], [1, 0]],
        [[-1, -1], [1, 0], [1, 1]],
        [[-1, -1], [1, 0], [1, 2]],
        [[-1, -1], [1, 1], [1, 2]],
        [[-1, -1], [-1, -1], [1, 2]],
    ]
    # Perform the step
    rewards = []
    is_terminated = False
    is_truncated = False
    for i, action in enumerate(actions):
        schedule_buffer = scheduler.schedule_buffer.copy()
        # Assert that schedule_buffer matches expected values
        assert (
            schedule_buffer == expected_schedule_buffer[i]
        ), "Schedule buffer is incorrect after the action."

        # Updated ETD values after one action
        assert [
            job_list[0].estimated_tardiness for job_list in scheduler.jobs
        ] == expected_initial_etd[
            i
        ], "ETD values did successfully update after the action."

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        is_terminated = terminated
        is_truncated = truncated

    # Final Reward
    assert rewards[-1] == -17.77777777777778, "Final reward is incorrect."

    # Final Flag
    assert is_terminated == True, "Environment is not terminated."
    assert is_truncated == False, "Environment is truncated."
