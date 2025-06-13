import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=3):
        self.size = size
        self.state = 0 
        self.goal = size * size - 1
        self.actions = [0, 1, 2, 3]
        self.n_states = size * size
        self.n_actions = len(self.actions)
        
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        row = self.state // self.size
        col = self.state % self.size
        
        if action == 0:  # Up
            row = max(0, row - 1)
        elif action == 1:  # Right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # Down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # Left
            col = max(0, col - 1)
            
        self.state = row * self.size + col
        
        done = self.state == self.goal
        reward = 1 if done else -0.1
        
        return self.state, reward, done

class QLearning:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        # Q-learning learn (off-policy)
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

class SARSA:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, next_action):
        # SARSA learn (on-policy)
        td_target = reward + self.gamma * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

def train_agent(agent, env, episodes=1000):
    rewards_history = []
    
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        # For SARSA, we need to select the initial action before the loop
        # because we need both current and next actions to update Q-values
        if isinstance(agent, SARSA):
            action = agent.choose_action(state)
            
        while not done:
            # For Q-learning, we can select action inside the loop
            # because we only need current state and action to update
            if isinstance(agent, QLearning):
                action = agent.choose_action(state)
            
            next_state, reward, done = env.step(env.actions[action])
            total_reward += reward
            
            if isinstance(agent, SARSA):
                # SARSA needs to know the next action to update Q-values
                # This is why it's on-policy - it uses the actual next action
                next_action = agent.choose_action(next_state)
                agent.learn(state, action, reward, next_state, next_action)
                action = next_action
            else:  # Q-learning
                # Q-learning is off-policy - it uses max Q-value of next state
                # regardless of what action will actually be taken
                agent.learn(state, action, reward, next_state)
            
            state = next_state
            
        rewards_history.append(total_reward)
    
    return rewards_history

def plot_results(q_learning_rewards, sarsa_rewards):
    # Calculate moving average over 10 epochs
    window_size = 10
    q_learning_avg = np.convolve(q_learning_rewards, np.ones(window_size)/window_size, mode='valid')
    sarsa_avg = np.convolve(sarsa_rewards, np.ones(window_size)/window_size, mode='valid')
    
    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Q-learning plot
    # ax1.plot(q_learning_rewards, label='Raw rewards', alpha=0.1, color='blue')
    ax1.plot(q_learning_avg, label='10-epoch average', color='blue', linewidth=2)
    ax1.set_title('Q-learning Performance')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    ax1.legend()
    
    # SARSA plot
    # ax2.plot(sarsa_rewards, label='Raw rewards', alpha=0.1, color='orange')
    ax2.plot(sarsa_avg, label='10-epoch average', color='orange', linewidth=2)
    ax2.set_title('SARSA Performance')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Total Reward')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_policy_heatmaps(q_learning_agent, sarsa_agent, env):
    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Function to get policy visualization
    def get_policy_heatmap(agent):
        policy = np.zeros((env.size, env.size))
        for state in range(env.n_states):
            row = state // env.size
            col = state % env.size
            # Get the best action for this state
            best_action = np.argmax(agent.q_table[state])
            # Map actions to values: up=0, right=1, down=2, left=3
            policy[row, col] = best_action
        return policy
    
    # Get policies
    q_learning_policy = get_policy_heatmap(q_learning_agent)
    sarsa_policy = get_policy_heatmap(sarsa_agent)
    
    # Plot Q-learning policy
    im1 = ax1.imshow(q_learning_policy, cmap='viridis')
    ax1.set_title('Q-learning Policy')
    ax1.set_xticks(np.arange(env.size))
    ax1.set_yticks(np.arange(env.size))
    ax1.set_xticklabels(['0', '1', '2'])
    ax1.set_yticklabels(['0', '1', '2'])
    
    # Add action labels
    for i in range(env.size):
        for j in range(env.size):
            action = int(q_learning_policy[i, j])
            action_symbol = ['↑', '→', '↓', '←'][action]
            ax1.text(j, i, action_symbol, ha='center', va='center', color='white')
    
    # Plot SARSA policy
    im2 = ax2.imshow(sarsa_policy, cmap='viridis')
    ax2.set_title('SARSA Policy')
    ax2.set_xticks(np.arange(env.size))
    ax2.set_yticks(np.arange(env.size))
    ax2.set_xticklabels(['0', '1', '2'])
    ax2.set_yticklabels(['0', '1', '2'])
    
    # Add action labels
    for i in range(env.size):
        for j in range(env.size):
            action = int(sarsa_policy[i, j])
            action_symbol = ['↑', '→', '↓', '←'][action]
            ax2.text(j, i, action_symbol, ha='center', va='center', color='white')
    
    # Add colorbar
    plt.colorbar(im1, ax=ax1, label='Action (0:↑, 1:→, 2:↓, 3:←)')
    plt.colorbar(im2, ax=ax2, label='Action (0:↑, 1:→, 2:↓, 3:←)')
    
    plt.tight_layout()
    plt.show()

def train_and_plot():
    # Create environment
    env = GridWorld(size=3)
    
    # Initialize agents
    q_learning_agent = QLearning(env.n_states, env.n_actions)
    sarsa_agent = SARSA(env.n_states, env.n_actions)
    
    # Train agents
    print("Training Q-learning agent...")
    q_learning_rewards = train_agent(q_learning_agent, env)
    
    print("Training SARSA agent...")
    sarsa_rewards = train_agent(sarsa_agent, env)
    
    # Plot learning curves
    plot_results(q_learning_rewards, sarsa_rewards)
    
    # Plot final policies
    plot_policy_heatmaps(q_learning_agent, sarsa_agent, env)
    
    # Print final Q-tables
    print("\nQ-learning final Q-table:")
    print(q_learning_agent.q_table)
    print("\nSARSA final Q-table:")
    print(sarsa_agent.q_table)

if __name__ == "__main__":
    train_and_plot()
