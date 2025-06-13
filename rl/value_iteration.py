import numpy as np

GRID_SIZE = 4
GAMMA = 0.99
THETA = 1e-3

V = np.zeros((GRID_SIZE, GRID_SIZE))
policy = np.full((GRID_SIZE, GRID_SIZE), 'U')
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

def value_iteration():
    iteration = 0
    while True:
        iteration += 1
        print(f"\nIteration {iteration}")
        
        delta = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i, j) == (3, 3):
                    continue
                
                v = V[i, j]
                action_values = {}
                for a in all_actions:
                    (ni, nj), r = step((i, j), a)
                    action_values[a] = r + GAMMA * V[ni, nj]

                V[i, j] = max(action_values.values())
                delta = max(delta, abs(v - V[i, j]))
        
        print("\nValue function:")
        print(V)
        
        if delta < THETA:
            break
        
        if iteration > 100:
            print("Warning: Maximum iterations reached")
            break

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) == (3, 3):
                continue
            
            action_values = {}
            for a in all_actions:
                (ni, nj), r = step((i, j), a)
                action_values[a] = r + GAMMA * V[ni, nj]
            best_action = max(action_values, key=action_values.get)
            policy[i, j] = best_action
    
    print("\nFinal Policy:")
    print(policy)
    return V, policy

if __name__ == "__main__":
    V, policy = value_iteration()
