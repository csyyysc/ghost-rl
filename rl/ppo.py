import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque


class ActorCritic(nn.Module):
    
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Image processing (CarRacing has 96x96x3 input)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_output_size(observation_space.shape)
        
        # Shared feature layers
        self.shared_layers = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_space.shape[0])
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Action bounds for continuous actions
        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)
    
    def _get_conv_output_size(self, input_shape):
        """Calculate the output size of convolutional layers"""
        
        dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.size()))
    
    def forward(self, state):
        conv_out = self.conv_layers(state)
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        features = self.shared_layers(conv_out)
        
        action_mean = self.actor(features)
        action_mean = torch.tanh(action_mean)
        action_mean = action_mean * self.action_scale + self.action_bias
        
        value = self.critic(features)
        
        return action_mean, value
    
    def get_action_and_value(self, state, action=None):
        """Get action and value with log probabilities"""
        action_mean, value = self.forward(state)
        
        # Create action distribution (assuming continuous actions with fixed std)
        action_std = torch.ones_like(action_mean) * 0.5
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = action_dist.sample()
        
        action_logprob = action_dist.log_prob(action).sum(axis=-1)
        entropy = action_dist.entropy().sum(axis=-1)
        
        return action, action_logprob, entropy, value


class PPOAgent:
    
    def __init__(self, env, lr=3e-4, gamma=0.99, clip_ratio=0.2, 
                 update_epochs=10, batch_size=64, buffer_size=2048):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.actor_critic = ActorCritic(env.observation_space, env.action_space).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def preprocess_state(self, state):
        """Preprocess the state (normalize and convert to tensor)"""

        if isinstance(state, tuple):
            state = state[0]  # Handle gym environments that return (obs, info)
        
        state = np.transpose(state, (2, 0, 1))  # Change from HWC to CHW
        state = state / 255.0
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def collect_experience(self, num_steps):
        """Collect experience for training"""

        state, _ = self.env.reset()
        state = self.preprocess_state(state)
        
        episode_reward = 0
        episode_length = 0
        
        for _ in range(num_steps):
            with torch.no_grad():
                action, logprob, _, value = self.actor_critic.get_action_and_value(state)
            
            next_state, reward, done, truncated, _ = self.env.step(action.cpu().numpy()[0])
            
            self.states.append(state)
            self.actions.append(action)
            self.logprobs.append(logprob)
            self.rewards.append(reward)
            self.values.append(value)
            self.dones.append(done or truncated)
            
            episode_reward += reward
            episode_length += 1
            
            state = self.preprocess_state(next_state)
            
            if done or truncated:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                state, _ = self.env.reset()
                state = self.preprocess_state(state)
                episode_reward = 0
                episode_length = 0
    
    def compute_advantages(self):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""

        gae_lambda = 0.95
        
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        values = torch.cat(self.values).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        next_value = 0 if self.dones[-1] else values[-1].item()
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self):

        if len(self.states) < self.batch_size:
            return {}
        
        advantages, returns = self.compute_advantages()
        
        old_states = torch.cat(self.states).to(self.device)
        old_actions = torch.cat(self.actions).to(self.device)
        old_logprobs = torch.cat(self.logprobs).to(self.device)

        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for _ in range(self.update_epochs):
            indices = torch.randperm(len(old_states))
            
            for start in range(0, len(old_states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                
                _, new_logprobs, entropy, new_values = self.actor_critic.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_logprobs - batch_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
                
                # Entropy loss (to encourage exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        self.clear_storage()
        
        return {
            'policy_loss': total_policy_loss / self.update_epochs,
            'value_loss': total_value_loss / self.update_epochs,
            'entropy_loss': total_entropy_loss / self.update_epochs
        }
    
    def clear_storage(self):
        """Clear experience storage"""

        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def save_model(self, filepath):

        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath):
        """Load the model"""

        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train():
    """Train PPO agent on CarRacing environment"""

    env = gym.make('CarRacing-v3', continuous=True, render_mode=None)
    agent = PPOAgent(env)
    
    total_timesteps = 1000000
    update_frequency = 2048
    
    print(f"Training PPO on CarRacing-v3")
    print(f"Device: {agent.device}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Update frequency: {update_frequency}")
    print("-" * 50)
    
    timestep = 0
    update_count = 0
    
    rewards = []
    losses = []
    
    while timestep < total_timesteps:
        agent.collect_experience(update_frequency)
        timestep += update_frequency
        
        losses = agent.update_policy()
        update_count += 1
        
        if len(agent.episode_rewards) > 0:
            avg_reward = np.mean(agent.episode_rewards)
            avg_length = np.mean(agent.episode_lengths)
            rewards.append(avg_reward)
            
            if losses:
                losses.append(losses)
                print(f"Update {update_count} | Timestep {timestep}")
                print(f"Average Reward: {avg_reward:.2f} | Average Length: {avg_length:.1f}")
                print(f"Policy Loss: {losses['policy_loss']:.4f} | Value Loss: {losses['value_loss']:.4f}")
                print("-" * 50)
        
        if update_count % 10 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            agent.save_model(f"ppo_model_{timestamp}.pt")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.save_model(f"ppo_model_final_{timestamp}.pt")
    
    if rewards:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('Average Episode Reward')
        plt.xlabel('Update')
        plt.ylabel('Reward')
        
        if losses:
            plt.subplot(1, 2, 2)
            policy_losses = [loss['policy_loss'] for loss in losses]
            value_losses = [loss['value_loss'] for loss in losses]
            plt.plot(policy_losses, label='Policy Loss')
            plt.plot(value_losses, label='Value Loss')
            plt.title('Training Losses')
            plt.xlabel('Update')
            plt.ylabel('Loss')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'ppo_training_progress_{timestamp}.png')
        plt.show()
    
    env.close()
    return agent


def simulate(model_path):
    """Simulate trained PPO agent"""

    env = gym.make('CarRacing-v3', continuous=True, render_mode='human')
    agent = PPOAgent(env)
    agent.load_model(model_path)
    
    num_episodes = 5
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = agent.preprocess_state(state)
        
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.actor_critic.get_action_and_value(state)
            
            next_state, reward, done, truncated, _ = env.step(action.cpu().numpy()[0])
            state = agent.preprocess_state(next_state)
            
            episode_reward += reward
            done = done or truncated
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()


if __name__ == "__main__":
    # print("CarRacing-v3 (PPO) training...")
    # agent = train()
    
    # Uncomment to test a trained model
    simulate("ppo_model_20250614_170429.pt")
