import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

torch.manual_seed(42)
np.random.seed(42)


class Actor(nn.Module):
    """Actor network that outputs action parameters for continuous action space"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Better initialization for more stable learning
        with torch.no_grad():
            # Initialize final layer weights and biases
            self.actor_net[-2].weight.data.uniform_(-0.1, 0.1)
            self.actor_net[-2].bias.data.uniform_(-0.1, 0.1)

            # Initialize log_std to promote exploration
            self.log_std.weight.data.uniform_(-0.1, 0.1)
            self.log_std.bias.data.uniform_(-1.0, -0.5)

    def forward(self, state):
        mean = self.actor_net(state)
        mean = mean * 2  # Scale to [-2, 2] for Pendulum (non-in-place)

        # Get intermediate representation for log_std
        x = F.relu(state @ self.actor_net[0].weight.T + self.actor_net[0].bias)
        x = F.relu(x @ self.actor_net[2].weight.T + self.actor_net[2].bias)
        log_std = self.log_std(x)

        # Clamp to reasonable range =>
        #   Extremely small or large standard deviations can lead to exploding or vanishing gradients when training.
        # exp(-20) ~ 2e-9
        # exp(0) = 1
        log_std = torch.clamp(log_std, -20, 0)
        return mean, log_std

    def get_action(self, state):
        """Sample action from the policy"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action = torch.clamp(action, -2.0, 2.0)
        return action.detach(), log_prob.detach()


class Critic(nn.Module):
    """Critic network that estimates state value function V(s)"""

    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.value_net(state)


class ActorCritic:
    """Actor-Critic algorithm implementation"""

    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=3e-4, gamma=0.99):
        self.gamma = gamma

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr_critic)

        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

    def normalize_rewards(self, rewards):
        """Normalize rewards using running statistics"""

        rewards = np.array(rewards)

        # Update running statistics (Welford's algorithm)
        for reward in rewards:
            self.reward_count += 1
            delta = reward - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = reward - self.reward_mean
            self.reward_std = np.sqrt(max(1e-8,
                                          ((self.reward_count - 2) * self.reward_std**2 + delta * delta2) / (self.reward_count - 1)))

        normalized_rewards = (rewards - self.reward_mean) / \
            (self.reward_std + 1e-8)
        return normalized_rewards.tolist()

    def update(self, states, actions, rewards, next_states, dones):
        """Update both actor and critic networks"""

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))

        # Normalize rewards for better learning
        normalized_rewards = self.normalize_rewards(rewards)
        rewards = torch.FloatTensor(normalized_rewards)

        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(np.array(dones))

        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()

        td_targets = rewards + self.gamma * next_values * (~dones)
        advantages = td_targets - values

        # CRITICAL: Normalize advantages for stable learning
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        for _ in range(2):
            critic_loss = F.mse_loss(values, td_targets.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

            values = self.critic(states).squeeze()

        mean, log_std = self.actor(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Add entropy for exploration
        entropy = dist.entropy().sum(dim=-1).mean()
        actor_loss = -(log_probs * advantages.detach()).mean() - 0.01 * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()


def main():
    """Main training function"""
    # Environment setup
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize agent
    agent = ActorCritic(state_dim, action_dim)

    # Training parameters
    num_episodes = 5000
    max_steps = 200
    update_frequency = 64  # Less frequent updates for stability

    # Tracking variables
    episode_rewards = []
    critic_losses = []
    actor_losses = []
    recent_rewards = deque(maxlen=100)

    # Storage for batch updates
    states_batch = []
    actions_batch = []
    rewards_batch = []
    next_states_batch = []
    dones_batch = []

    print("Starting Actor-Critic training on Pendulum-v1...")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print("Key improvements: Advantage normalization, reward normalization, gradient clipping, entropy bonus")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            # Get action from actor

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = agent.actor.get_action(state_tensor)
            action_np = action.numpy().flatten()

            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # Store experience
            states_batch.append(state)
            actions_batch.append(action_np)
            rewards_batch.append(reward)
            next_states_batch.append(next_state)
            dones_batch.append(done)

            episode_reward += reward
            state = next_state

            # Update networks when batch is full
            if len(states_batch) >= update_frequency:
                actor_loss, critic_loss = agent.update(
                    states_batch, actions_batch, rewards_batch,
                    next_states_batch, dones_batch
                )
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                # Clear batch
                states_batch.clear()
                actions_batch.clear()
                rewards_batch.clear()
                next_states_batch.clear()
                dones_batch.clear()

            if done:
                break

        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(recent_rewards)
            print(
                f"Episode {episode + 1}, Average Reward (last 100): {avg_reward:.2f}")

    env.close()
    return episode_rewards, critic_losses, actor_losses


def plot_results(episode_rewards, critic_losses):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(episode_rewards, alpha=0.7, color='blue', linewidth=1)

    window_size = 50
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(
            window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg,
                 color='red', linewidth=2, label=f'Moving Average ({window_size} episodes)')
        ax1.legend()

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards Over Time')
    ax1.grid(True, alpha=0.3)

    if critic_losses:
        ax2.plot(critic_losses, alpha=0.7, color='green', linewidth=1)
        loss_window = min(100, len(critic_losses) // 10)
        if len(critic_losses) >= loss_window and loss_window > 1:
            loss_moving_avg = np.convolve(critic_losses, np.ones(
                loss_window)/loss_window, mode='valid')
            ax2.plot(range(loss_window-1, len(critic_losses)), loss_moving_avg,
                     color='red', linewidth=2, label=f'Moving Average ({loss_window} updates)')
            ax2.legend()

    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('Critic Loss')
    ax2.set_title('Critic Loss Over Training')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot to current working directory
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'actor_critic_training_results_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")

    plt.show()

    print(f"\nTraining completed!")
    print(
        f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    if critic_losses:
        print(f"Final critic loss: {critic_losses[-1]:.4f}")


def test_trained_agent():
    """Test the trained agent and visualize performance"""
    env = gym.make('Pendulum-v1', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ActorCritic(state_dim, action_dim)

    print("\nTesting agent performance...")
    test_rewards = []

    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(200):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent.actor(state_tensor)
                action = mean.numpy().flatten()  # Use mean action (no exploration)

            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}")

    env.close()
    print(f"Average test reward: {np.mean(test_rewards):.2f}")


if __name__ == "__main__":
    episode_rewards, critic_losses, actor_losses = main()

    plot_results(episode_rewards, critic_losses)

    # test_trained_agent()
