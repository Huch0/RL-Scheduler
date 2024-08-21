from train.test_model import test_model
from train.make_env import make_env
from sb3_contrib import MaskablePPO

import numpy as np
import copy

def ensemble_predict(models, obs, action_masks):
    # 각 모델의 예측 결과를 튜플로 변환하여 다수결 방식 적용
    # action_probs = [tuple(model.policy.predict(obs, deterministic=True, action_masks=action_masks)[0]) for model in models]
    # # 다수결로 행동을 선택하는 방식
    # return max(set(action_probs), key=action_probs.count)

    # 혹은, 평균을 계산하는 방법으로:
    action_probs = np.mean([model.policy.predict(obs, deterministic=True, action_masks=action_masks)[0] for model in models], axis=0)

    return np.argmax(action_probs)

repeats = [
    [4, 3, 3, 5, 4, 2, 3, 2, 2, 3, 3, 4],
    [3, 3, 3, 3, 4, 2, 3, 2, 1, 3, 3, 2],
    [5, 1, 3, 2, 4, 4, 3, 3, 2, 1, 2, 3],
    [4, 4, 2, 2, 1, 1, 1, 4, 2, 2, 1, 3],
    [1, 2, 2, 3, 2, 1, 2, 3, 3, 3, 2, 2],
    [2, 2, 2, 1, 3, 2, 1, 3, 2, 3, 3, 3],
    [4, 1, 3, 2, 2, 2, 2, 3, 1, 3, 3, 1],
    [4, 4, 4, 2, 1, 4, 2, 4, 3, 3, 3, 3],
    [3, 4, 3, 3, 4, 1, 1, 3, 1, 4, 2, 2],
    [4, 4, 4, 3, 2, 4, 2, 3, 3, 2, 3, 3]
]

LWKR_MOD = [1891.0, 1313.0, 1415.0, 923.0, 1013.0, 992.0, 979.0, 1880.0, 1245.0, 2033.0]

cost_list = [5, 1, 2, 10]
profit_per_time = 10
max_time = 50


params = {
    "policy_kwargs": dict(
        net_arch=[256, 128, 64]
    ),
    "learning_rate": 0.00005,
}
model_path = "MP_env_test1v2"
# model = MaskablePPO.load("./models/Env4/MP_Multi_Env4_binary_heatmap_v4_lr_1e-05", **params)
model = MaskablePPO.load(model_path, **params)

total_cost_results = []

for repeat in repeats:
    if repeat != [4, 3, 3, 5, 4, 2, 3, 2, 2, 3, 3, 4]:
        continue
    total_cost = 10000
    for _ in range(1):
        env, _ = make_env(num_machines = 8, num_jobs = 12, max_repeats = 12, repeat_means = repeat, repeat_stds = [1] * 12, test_mode = True, cost_list = cost_list, profit_per_time = profit_per_time, max_time = max_time)
        info = test_model(env=env, model=model, deterministic=True, render=False, random_simulation = False, debug_step=[100, 105])
        total_cost = min(total_cost, info["cost_deadline"] + info["cost_hole"] + info["cost_makespan"] + info["cost_processing"])
        # total_cost += info["cost_deadline"] + info["cost_hole"] + info["cost_makespan"] + info["cost_processing"]
    
    # total_cost /= 10
    total_cost_results.append(total_cost)

for i, total_cost in enumerate(total_cost_results):
    print(f"test case {i+1} : {total_cost} \t/ LWKR+MOD {LWKR_MOD[i]}", end = "\t")
    if total_cost < LWKR_MOD[i]:
        print("WIN")
    else:
        print("LOSE")