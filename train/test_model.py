# import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch
import numpy as np

def random_tardiness_simulation(test_env, repeats):
    j = []
    for r in repeats:
        j.append((r, 1))
    
    test_env.set_test_mode(True)

    step = 0
    obs, info = test_env.reset()
    while True:
        step += 1
        action = test_env.action_space.sample()    
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        info["reward"] = reward
        info["env"] = test_env
        info["profit_ratio"] = test_env.profit_per_time
        
        if done:
            test_env.print_result(info)
            test_env.render()
            break

def plot_policy_function(model, obs, action_masks):
    """
    모델의 정책을 시각화하는 함수입니다.
    
    Args:
    - model: MaskablePPO 모델
    - obs: 현재 상태(Observation)
    - action_masks: 행동 마스크(Action Masks)
    """
    obs_tensor = convert_obs_to_tensor(obs)
    action_prob = model.policy.get_distribution(obs=obs_tensor, action_masks=action_masks)
    action_probs = action_prob.distribution.probs.detach().numpy()

    # Action 번호 생성 및 마스크 적용
    valid_actions = np.where(action_masks == 1)[0]
    valid_action_probs = action_probs[0, valid_actions]  # 실행 가능한 액션들에 대한 확률 선택

    # 2차원으로 변환 (8x12)
    action_probs_2d = np.zeros((8, 12))  # 0으로 초기화된 8x12 배열 생성
    for idx, action in enumerate(valid_actions):
        row = action // 12  # machine (row)
        col = action % 12   # job (col)
        action_probs_2d[row, col] = valid_action_probs[idx]

    # 최선의 행동 찾기 (최대 확률을 가진 행동의 위치)
    best_action_idx = np.unravel_index(np.argmax(action_probs_2d, axis=None), action_probs_2d.shape)
    best_action_prob = action_probs_2d[best_action_idx]

    # 최선의 행동 출력
    print(f"Machine : [{best_action_idx[0]}] <-- Job : [{best_action_idx[1]+1}] is best ({best_action_prob*100:.2f}%)!")

    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(action_probs_2d, cmap='Reds', aspect='auto', vmin=0, vmax=1)  # vmin, vmax 설정
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)

    # Valid하지 않은 행동을 회색으로 표시
    for i in range(8):
        for j in range(12):
            action_index = i * 12 + j
            if action_masks[action_index] == 0:  # Valid하지 않은 경우
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color='gray', alpha=0.5))

    cbar = fig.colorbar(cax, ax=ax, label='Probability')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # 색상 바의 주요 구간 설정

    ax.set_xlabel('Job Index')
    ax.set_ylabel('Machine Index')
    ax.set_title('Valid Action Probability Distribution (8x12)')
    plt.show()

def convert_obs_to_tensor(obs):
    """
    Convert a dictionary of numpy arrays to a dictionary of PyTorch tensors.

    Args:
    - obs (Dict[str, np.ndarray]): The input observation dictionary.

    Returns:
    - Dict[str, torch.Tensor]: The converted dictionary with tensors.
    """
    
    obs_tensor = {}
    for key, value in obs.items():
        # 만약 key가 schedule_heatmap인 경우 1차원 텐서로 변환
        if key == "schedule_heatmap":
            obs_tensor[key] = torch.tensor(value, dtype=torch.float32).flatten()
            continue
        # 각 요소의 차원과 타입을 확인하여 적절히 변환
        try:
            obs_tensor[key] = torch.tensor(value, dtype=torch.float32)
        except Exception as e:
            print(f"Error converting {key}: {e}")
            obs_tensor[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)

    # 모든 key에 대하여 tensor의 크기 확인
    # for key, tensor in obs_tensor.items():
    #     print(f"Key: {key}, Size: {tensor.size()}")

    for key, tensor in obs_tensor.items():
        if tensor.dim() == 1:
            obs_tensor[key] = tensor.unsqueeze(0)  # 1차원 텐서를 2차원으로 변환
    return obs_tensor


def test_model(env, model, detail_mode = False, deterministic = True, render = True, random_simulation = False, debug_step = None):
    obs, info = env.reset()
    step = 0
    
    while True:
        action_masks = env.action_masks()
        if debug_step is not None:
            if type(debug_step) is list:
                if debug_step[0] <= step <= debug_step[1]:
                    env.render()
                    plot_policy_function(model, obs, action_masks)
            elif step == debug_step:
                env.render()
                plot_policy_function(model, obs, action_masks)

        action, _states = model.predict(obs, action_masks=action_masks, deterministic = deterministic)
        
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated

        if done:    
            info["reward"] = reward
            info["env"] = env
            info["profit_ratio"] = env.profit_per_time

            if render:
                env.print_result(info, detail_mode = detail_mode)
                env.render()
            break
    
    if random_simulation:
        print()
        print()
        print()
        print("---------------------------Random Simmulation---------------------------")
        random_tardiness_simulation(env, info["current_repeats"])

    return info