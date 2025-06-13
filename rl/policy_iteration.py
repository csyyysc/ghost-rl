import numpy as np

GRID_SIZE = 4
GAMMA = 0.99
THETA = 1e-3

V = np.zeros((GRID_SIZE, GRID_SIZE))

policy = np.full((GRID_SIZE, GRID_SIZE), 'U')
# Terminal state has no policy
policy[3, 3] = 'X'

actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
all_actions = list(actions.keys())

def step(state, action):
    if state == (3, 3):
        return state, 10
    act = actions[action]
    next_state = (min(max(state[0] + act[0], 0), 3), min(max(state[1] + act[1], 0), 3))
    reward = -1
    return next_state, reward

def policy_iteration():
    iteration = 0
    is_stable = False
    while not is_stable:
        iteration += 1
        print(f"\nIteration {iteration}")
        
        # Policy Evaluation
        eval_iter = 0
        while True:
            delta = 0
            eval_iter += 1
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    if (i, j) == (3, 3):
                        continue
                    v = V[i, j]
                    a = policy[i, j]
                    (ni, nj), r = step((i, j), a)
                    V[i, j] = r + GAMMA * V[ni, nj]
                    delta = max(delta, abs(v - V[i, j]))
            if delta < THETA:
                break
            if eval_iter % 10 == 0:
                # print(f"Policy evaluation iteration {eval_iter}, delta: {delta}")
                pass
        
        print("\nValue function:")
        print(V)
        print("\nPolicy(Before):")
        print(policy)
        
        # Policy Improvement
        is_stable = True
        policy_changes = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i, j) == (3, 3):  # Skip terminal state
                    continue
                old_action = policy[i, j]
                action_values = {}
                for a in all_actions:
                    (ni, nj), r = step((i, j), a)
                    action_values[a] = r + GAMMA * V[ni, nj]
                best_action = max(action_values, key=action_values.get)
                policy[i, j] = best_action
                if best_action != old_action:
                    is_stable = False
                    policy_changes += 1
        
        print(f"\nPolicy changes in this iteration: {policy_changes}")

        print("\nPolicy(After):")
        print(policy)
    
        if iteration > 100:  # Safety check to prevent infinite loops
            print("Warning: Maximum iterations reached")
            break
        
        print("-" * 100)