import time
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.env = env
        self.lr = learning_rate  
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        
    def _choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max * (1 - done))
        self.q_table[state, action] = new_value
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, episodes):
        rewards = []
        
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            print(f"-------------------- Episode {e} --------------------")
            
            while not done:
                action = self._choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")
                
                self.learn(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)
            
            if e % 100 == 0:
                print(f"Episode {e}, Average Reward: {np.mean(rewards[-100:]):.2f}, Epsilon: {self.epsilon:.2f}")
        
        return rewards
    
    def play_episode(self):
        state, _ = self.env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = np.argmax(self.q_table[state])
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            self.env.render()
            time.sleep(0.5)
        
        return total_reward

    def save_model(self, filename='q_table.npy'):
        np.save(filename, self.q_table)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='q_table.npy'):
        try:
            self.q_table = np.load(filename)
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"No saved model found at {filename}")
            return False

def plot_rewards(rewards):
    window_size = 100
    avg_rewards = []
    for i in range(0, len(rewards), window_size):
        window = rewards[i:i + window_size]
        if len(window) == window_size:
            avg_rewards.append(np.mean(window))
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(window_size, len(rewards) + 1, window_size), avg_rewards, 'b-')
    plt.title('Average Rewards per 100 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.show()

def get_env_info(env):
    print(f"Observation space number of states: {env.observation_space.n}")
    print(f"Action space number of actions: {env.action_space.n}")

def simulate(agent, episodes=5):
    print("\nPlaying episodes with trained agent:")

    for i in range(episodes):
        print(f"\nEpisode {i+1}")
        total_reward = agent.play_episode()
        print(f"Total reward: {total_reward}")

def main(args):
    if args.render:
        env = gym.make('FrozenLake-v1', render_mode="human")
    else:
        env = gym.make('FrozenLake-v1')

    get_env_info(env)

    agent = QLearningAgent(env)
    
    if args.render:
        # Try to load existing model
        if not agent.load_model():
            print("No saved model found. Please train the model first.")
            return
    else:
        # Train and save the model
        rewards = agent.train(episodes=args.episodes)
        agent.save_model()
        plot_rewards(rewards)

    if args.render:
        simulate(agent, args.render_episodes)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Q-Learning DQN')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to simulate')
    parser.add_argument('--render_episodes', type=int, default=5, help='Number of episodes to render')
    parser.add_argument('--model_path', type=str, default='q_table.npy', help='Path to save/load the model')
    args = parser.parse_args()

    main(args)
