import pickle
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

class REINFORCE:
    def __init__(self, env_name, learning_rate=0.01, gamma=0.99, render_mode=None):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        self.policy = PolicyNetwork(self.input_dim, self.output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        
    def train_episode(self):
        state, _ = self.env.reset()
        log_probs = []
        rewards = []
        done = False
        truncated = False
        
        while not (done or truncated):
            action, log_prob = self.policy.get_action(state)
            state, reward, done, truncated, _ = self.env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            
        return log_probs, rewards
    
    def update_policy(self, log_probs, rewards):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
    def train(self, num_episodes):
        episode_rewards = []
        for episode in range(num_episodes):
            log_probs, rewards = self.train_episode()
            self.update_policy(log_probs, rewards)
            
            total_reward = sum(rewards)
            episode_rewards.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        
        return episode_rewards
    
    def evaluate(self, num_episodes=10, render=False):
        if render:
            eval_env = gym.make(self.env.spec.id, render_mode="human")
        else:
            eval_env = self.env
            
        total_rewards = []
        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action, _ = self.policy.get_action(state)
                state, reward, done, truncated, _ = eval_env.step(action)
                episode_reward += reward
                
            total_rewards.append(episode_reward)
            
        if render:
            eval_env.close()
            
        avg_reward = sum(total_rewards) / num_episodes
        print(f"Average reward over {num_episodes} episodes: {avg_reward}")
        return avg_reward
    
    @staticmethod
    def plot_learning_curve(rewards, filename='learning_curve.png'):
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title('Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig(filename)
        plt.close()
        print(f"Learning curve saved to {filename}")

        
    def save_model(self, filename='reinforce.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.policy, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename='reinforce.pkl'):
        with open(filename, 'rb') as f:
            self.policy = pickle.load(f)
        print(f"Model loaded from {filename}")
    

if __name__ == "__main__":
    agent = REINFORCE("CartPole-v1")
    agent.load_model()
    rewards = agent.train(num_episodes=1000)
    agent.plot_learning_curve(rewards)
    agent.evaluate(render=True)
    # agent.save_model()