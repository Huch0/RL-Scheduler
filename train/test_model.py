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

def test_model(env, model, detail_mode = False, deterministic = True, render = True, random_simulation = False):
    obs, info = env.reset()
    
    while True:
        action_masks = env.action_masks()
        action, _states = model.predict(obs, action_masks=action_masks, deterministic = deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
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