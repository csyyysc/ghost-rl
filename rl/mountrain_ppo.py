import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt


class PPONetwork(nn.Module):
    """Neural network for PPO with shared layers for policy and value function"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PPONetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy head (actor)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Value head (critic)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        shared_out = self.shared_layers(state)
        policy_logits = self.policy_head(shared_out)
        value = self.value_head(shared_out)
        return policy_logits, value

    def get_action_and_value(self, state):
        policy_logits, value = self.forward(state)
        policy_dist = torch.distributions.Categorical(logits=policy_logits)
        action = policy_dist.sample()
        log_prob = policy_dist.log_prob(action)
        return action.item(), log_prob, value.squeeze()

    def evaluate_actions(self, states, actions):
        policy_logits, values = self.forward(states)
        policy_dist = torch.distributions.Categorical(logits=policy_logits)
        log_probs = policy_dist.log_prob(actions)
        entropy = policy_dist.entropy()
        return log_probs, values.squeeze(), entropy


class PPOAgent:
    """PPO Agent for discrete action spaces"""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2,
                 k_epochs=4, entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Initialize network and optimizer
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Storage for trajectory data
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, value = self.network.get_action_and_value(
            state_tensor)

        # Store trajectory data (detach tensors to avoid gradient issues)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())

        return action

    def store_reward_done(self, reward, done):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_and_advantages(self, next_value=0):
        """Compute returns and advantages using simple TD returns"""
        returns = []
        advantages = []

        # Convert to tensors
        rewards = torch.FloatTensor(self.rewards)
        values = torch.stack(self.values) if self.values else torch.tensor([])
        dones = torch.FloatTensor(self.dones)

        if len(values) == 0:
            return torch.tensor([]), torch.tensor([])

        # Compute returns
        R = next_value
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        advantages = returns - values

        return returns, advantages

    def update(self):
        """Update the policy using PPO"""
        if len(self.states) == 0:
            print("Warning: No data to update with")
            return

        print(f"Updating with {len(self.states)} samples")

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()

        if len(advantages) == 0:
            print("Warning: No advantages computed")
            return

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        # Convert data to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.stack(self.log_probs)

        # Check policy entropy before update
        with torch.no_grad():
            old_policy_logits, _ = self.network(states)
            old_policy_dist = torch.distributions.Categorical(
                logits=old_policy_logits)
            old_entropy = old_policy_dist.entropy().mean()

        print(f"Batch stats: Mean advantage: {advantages.mean():.4f}, "
              f"Std advantage: {advantages.std():.4f}, "
              f"Policy entropy: {old_entropy:.4f}")

        # PPO update for k epochs
        for epoch in range(self.k_epochs):
            # Evaluate actions with current policy
            log_probs, values, entropy = self.network.evaluate_actions(
                states, actions)

            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages

            # Compute losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropy.mean()

            # Total loss
            total_loss = policy_loss + self.value_coef * \
                value_loss + self.entropy_coef * entropy_loss

            if epoch == 0:
                print(f"Losses - Policy: {policy_loss:.4f}, Value: {value_loss:.4f}, "
                      f"Entropy: {entropy_loss:.4f}, Total: {total_loss:.4f}")
                print(f"Ratio stats - Mean: {ratio.mean():.4f}, "
                      f"Min: {ratio.min():.4f}, Max: {ratio.max():.4f}")

            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Clear trajectory data
        self.clear_memory()

    def clear_memory(self):
        """Clear stored trajectory data"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()


def train_ppo_mountain_car(max_episodes=20000, max_timesteps=200, batch_size=4096):
    """Train PPO agent on MountainCar-v0 environment"""

    # Create environment
    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: {env.action_space}")

    # Create PPO agent with standard hyperparameters
    agent = PPOAgent(state_dim, action_dim, lr=3e-4, entropy_coef=0.01)

    # Training variables
    episode_rewards = []
    episode_lengths = []
    running_reward = deque(maxlen=100)
    solved_threshold = -110  # MountainCar is considered solved at -110

    print("Starting training...")

    total_timesteps = 0
    episode = 0

    while episode < max_episodes:
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for timestep in range(max_timesteps):
            # Select action
            action = agent.select_action(state)

            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)

            # Simple reward shaping: small bonus for moving right
            position, velocity = next_state
            shaped_reward = reward + velocity * 0.1
            if velocity > 0:  # Moving right
                shaped_reward += 0.1

            # Store reward and done
            agent.store_reward_done(shaped_reward, done or truncated)

            # Update counters
            episode_reward += reward  # Keep original reward for logging
            episode_length += 1
            total_timesteps += 1
            state = next_state

            # Update agent when we have enough samples
            if total_timesteps % batch_size == 0:
                print(f"\n--- Update at timestep {total_timesteps} ---")
                agent.update()

            if done or truncated:
                break

        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        running_reward.append(episode_reward)

        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(running_reward)
            avg_length = np.mean(list(running_reward)
                                 ) if running_reward else 200
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, "
                  f"Episode Reward: {episode_reward:.2f}, Episode Length: {episode_length}")

        # Check if solved
        if len(running_reward) >= 100 and np.mean(running_reward) >= solved_threshold:
            print(
                f"Solved! Average reward over last 100 episodes: {np.mean(running_reward):.2f}")
            break

        episode += 1

    # Final update with any remaining data
    if len(agent.states) > 0:
        print(f"\n--- Final update with {len(agent.states)} samples ---")
        agent.update()

    env.close()

    # Plot results
    plot_results(episode_rewards, episode_lengths)

    return agent, episode_rewards, episode_lengths


def plot_results(episode_rewards, episode_lengths):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot episode rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)

    # Plot moving average
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(
            episode_rewards, np.ones(100)/100, mode='valid')
        ax1.plot(range(99, len(episode_rewards)), moving_avg,
                 'r-', label='Moving Average (100 episodes)')
        ax1.legend()

    # Plot episode lengths
    ax2.plot(episode_lengths)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def test_agent(agent, num_episodes=5, render=False):
    """Test the trained agent"""
    env = gym.make('MountainCar-v0')

    test_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            if render:
                env.render()

            # Select action (no exploration)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                policy_logits, _ = agent.network(state_tensor)
                action = torch.argmax(policy_logits, dim=1).item()

            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1

            if done or truncated:
                break

        test_rewards.append(episode_reward)
        print(
            f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    env.close()
    print(f"Average test reward: {np.mean(test_rewards):.2f}")
    return test_rewards


if __name__ == "__main__":
    # Train the agent
    print("Training PPO agent on MountainCar-v0...")
    agent, rewards, lengths = train_ppo_mountain_car()

    # Test the trained agent
    print("\nTesting trained agent...")
    test_agent(agent, num_episodes=5)

    print("\nTraining completed!")
