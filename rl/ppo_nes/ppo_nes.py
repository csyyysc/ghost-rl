import ale_py
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from collections import deque


class AtariActorCritic(nn.Module):
    """Actor-Critic network specifically designed for Atari environments"""

    def __init__(self, observation_space, action_space, hidden_dim=512):
        super(AtariActorCritic, self).__init__()

        # Atari Assault has 210x160x3 input, but we'll resize to 84x84
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4,
                      padding=0),  # 4 frames stacked
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        # Calculate conv output size for 84x84x4 input
        conv_out_size = self._get_conv_output_size((4, 84, 84))

        # Shared feature layers
        self.shared_layers = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_space.n)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _get_conv_output_size(self, input_shape):
        """Calculate the output size of convolutional layers"""
        dummy_input = torch.zeros(1, *input_shape)
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.size()))

    def forward(self, state):
        conv_out = self.conv_layers(state)
        conv_out = conv_out.view(conv_out.size(0), -1)

        features = self.shared_layers(conv_out)

        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value

    def get_action_and_value(self, state, action=None):
        """Get action and value with log probabilities"""
        action_logits, value = self.forward(state)

        # Create action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)

        if action is None:
            action = action_dist.sample()

        action_logprob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action, action_logprob, entropy, value


class FrameStack:
    """Frame stacking for Atari environments"""

    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame):
        frame = self.preprocess_frame(frame)
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self.get_state()

    def step(self, frame):
        frame = self.preprocess_frame(frame)
        self.frames.append(frame)
        return self.get_state()

    def preprocess_frame(self, frame):
        """Preprocess Atari frame: grayscale, resize, normalize"""
        if isinstance(frame, tuple):
            frame = frame[0]

        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

        # Resize to 84x84
        frame = self.resize_frame(frame, (84, 84))

        # Normalize
        frame = frame / 255.0

        return frame

    def resize_frame(self, frame, size):
        """Simple resize implementation"""
        try:
            from scipy import ndimage
            return ndimage.zoom(frame, (size[0]/frame.shape[0], size[1]/frame.shape[1]))
        except ImportError:
            # Fallback to simple downsampling if scipy not available
            h, w = frame.shape
            new_h, new_w = size
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            resized = frame[::step_h, ::step_w]

            # Crop or pad to exact size
            if resized.shape[0] > new_h:
                resized = resized[:new_h, :]
            if resized.shape[1] > new_w:
                resized = resized[:, :new_w]

            # Pad if necessary
            if resized.shape[0] < new_h or resized.shape[1] < new_w:
                padded = np.zeros(size)
                padded[:resized.shape[0], :resized.shape[1]] = resized
                return padded

            return resized

    def get_state(self):
        return np.stack(list(self.frames), axis=0)


class NaturalEvolutionStrategy:
    """Natural Evolution Strategy for evolving policy parameters"""

    def __init__(self, param_count, population_size=50, sigma=0.1, learning_rate=0.01):
        self.param_count = param_count
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

        # Initialize population
        self.mu = np.zeros(param_count)  # Mean of search distribution
        self.best_params = None
        self.best_fitness = float('-inf')

    def generate_population(self):
        """Generate population of parameter vectors"""
        epsilon = np.random.randn(self.population_size, self.param_count)
        population = self.mu + self.sigma * epsilon
        return population, epsilon

    def update(self, fitnesses, epsilon):
        """Update the search distribution based on fitness scores"""
        # Rank-based fitness shaping
        sorted_indices = np.argsort(fitnesses)[::-1]  # Descending order

        # Use only top half of population
        elite_count = self.population_size // 2
        elite_indices = sorted_indices[:elite_count]

        # Compute weighted gradient
        weights = np.log(elite_count + 1) - \
            np.log(np.arange(1, elite_count + 1))
        weights = weights / np.sum(weights)

        weighted_epsilon = np.sum(
            weights.reshape(-1, 1) * epsilon[elite_indices], axis=0)

        # Update mean
        self.mu += self.learning_rate * weighted_epsilon

        # Track best parameters
        best_idx = sorted_indices[0]
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_params = self.mu + self.sigma * epsilon[best_idx]


class PPONESAgent:
    """Hybrid PPO-NES Agent for Atari environments"""

    def __init__(self, env_name="ALE/Assault-v5", lr=3e-4, gamma=0.99, clip_ratio=0.2,
                 update_epochs=4, batch_size=32, buffer_size=128,
                 nes_population_size=20, nes_sigma=0.1, nes_frequency=10):

        self.env_name = env_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create environment
        self.env = gym.make(env_name, render_mode=None)
        self.frame_stack = FrameStack(num_frames=4)

        # PPO parameters
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # NES parameters
        self.nes_frequency = nes_frequency
        self.generation = 0

        # Initialize network
        self.actor_critic = AtariActorCritic(
            self.env.observation_space,
            self.env.action_space
        ).to(self.device)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Initialize NES
        param_count = sum(p.numel()
                          for p in self.actor_critic.parameters() if p.requires_grad)
        self.nes = NaturalEvolutionStrategy(
            param_count,
            population_size=nes_population_size,
            sigma=nes_sigma
        )

        # Training storage
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

        # Metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_rewards = []
        self.nes_rewards = []

        # Persistent episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_active = False

        # State management
        self.current_state = None

    def parameters_to_vector(self):
        """Convert model parameters to a single vector"""
        params = []
        for param in self.actor_critic.parameters():
            if param.requires_grad:
                params.append(param.data.view(-1))
        return torch.cat(params).cpu().numpy()

    def vector_to_parameters(self, vector):
        """Load vector into model parameters"""
        vector = torch.FloatTensor(vector).to(self.device)
        pointer = 0
        for param in self.actor_critic.parameters():
            if param.requires_grad:
                num_param = param.numel()
                param.data = vector[pointer:pointer + num_param].view_as(param)
                pointer += num_param

    def evaluate_fitness(self, params, num_episodes=3):
        """Evaluate fitness of parameter vector"""
        # Save current parameters
        original_params = self.parameters_to_vector()

        # Load new parameters
        self.vector_to_parameters(params)

        total_reward = 0
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            state = self.frame_stack.reset(state)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            episode_reward = 0
            done = False
            max_steps = 1000  # Limit episode length for faster evaluation

            for step in range(max_steps):
                with torch.no_grad():
                    action, _, _, _ = self.actor_critic.get_action_and_value(
                        state)

                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.item())
                done = terminated or truncated

                state = self.frame_stack.step(next_state)
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                episode_reward += reward

                if done:
                    break

            total_reward += episode_reward

        # Restore original parameters
        self.vector_to_parameters(original_params)

        return total_reward / num_episodes

    def collect_experience(self, num_steps):
        """Collect experience for PPO training"""
        # Initialize or continue episode
        if not self.episode_active or self.current_state is None:
            state, _ = self.env.reset()
            state = self.frame_stack.reset(state)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.episode_active = True
            self.current_state = state
        else:
            # Continue from stored state
            state = self.current_state

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        for step in range(num_steps):
            with torch.no_grad():
                action, logprob, _, value = self.actor_critic.get_action_and_value(
                    state)

            next_state, reward, terminated, truncated, _ = self.env.step(
                action.item())
            done = terminated or truncated

            self.states.append(state)
            self.actions.append(action)
            self.logprobs.append(logprob)
            self.rewards.append(reward)
            self.values.append(value)
            self.dones.append(done)

            self.current_episode_reward += reward
            self.current_episode_length += 1

            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                print(
                    f"Episode completed: Length={self.current_episode_length}, Reward={self.current_episode_reward:.2f}")

                # Reset for new episode
                state, _ = self.env.reset()
                state = self.frame_stack.reset(state)
                self.current_episode_reward = 0
                self.current_episode_length = 0
                self.episode_active = True
                self.current_state = state
            else:
                state = self.frame_stack.step(next_state)
                self.current_state = state

            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def compute_advantages(self):
        """Compute advantages using GAE"""
        gae_lambda = 0.95

        rewards = torch.FloatTensor(self.rewards).to(self.device)
        values = torch.cat(self.values).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        next_value = 0
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * \
                next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        return advantages, returns

    def update_policy(self):
        """Update policy using PPO"""
        if len(self.states) < self.batch_size:
            return {}

        # Compute advantages
        advantages, returns = self.compute_advantages()

        # Prepare data
        states = torch.cat(self.states).to(self.device)
        actions = torch.cat(self.actions).to(self.device)
        old_logprobs = torch.cat(self.logprobs).to(self.device)

        # PPO update
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        entropy_loss_total = 0

        for epoch in range(self.update_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                _, new_logprobs, entropy, values = self.actor_critic.get_action_and_value(
                    batch_states, batch_actions
                )

                # Compute losses
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio,
                                    1 + self.clip_ratio) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_loss_total += entropy_loss.item()

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss_total,
            'value_loss': value_loss_total,
            'entropy_loss': entropy_loss_total
        }

    def nes_evolution_step(self):
        """Perform one NES evolution step"""
        print(f"Running NES evolution step (Generation {self.generation})")

        # Generate population
        population, epsilon = self.nes.generate_population()

        # Evaluate fitness for each individual
        fitnesses = []
        for i, individual in enumerate(population):
            fitness = self.evaluate_fitness(individual)
            fitnesses.append(fitness)
            print(
                f"Individual {i+1}/{len(population)}: Fitness = {fitness:.2f}")

        fitnesses = np.array(fitnesses)

        # Update NES
        self.nes.update(fitnesses, epsilon)

        # Load best parameters
        if self.nes.best_params is not None:
            self.vector_to_parameters(self.nes.best_params)
            print(f"Best fitness this generation: {np.max(fitnesses):.2f}")
            print(f"Best fitness overall: {self.nes.best_fitness:.2f}")

        self.nes_rewards.append(np.max(fitnesses))
        self.generation += 1

    def clear_storage(self):
        """Clear training storage"""
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def train(self, total_timesteps=100000, log_interval=10):
        """Train the agent using PPO with periodic NES evolution"""
        print(f"Starting PPO-NES training on {self.env_name}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"NES evolution every {self.nes_frequency} PPO updates")

        timestep = 0
        update_count = 0

        while timestep < total_timesteps:
            # Collect experience
            self.collect_experience(self.buffer_size)
            timestep += self.buffer_size

            # Update policy with PPO
            losses = self.update_policy()
            self.clear_storage()
            update_count += 1

            # Record average reward
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(list(self.episode_rewards))
                self.training_rewards.append(avg_reward)

            # Periodic NES evolution
            if update_count % self.nes_frequency == 0:
                self.nes_evolution_step()

            # Logging
            if update_count % log_interval == 0:
                avg_reward = np.mean(
                    list(self.episode_rewards)) if self.episode_rewards else 0
                avg_length = np.mean(
                    list(self.episode_lengths)) if self.episode_lengths else 0

                print(f"Update {update_count}, Timestep {timestep}")
                print(f"Episodes completed: {len(self.episode_rewards)}")
                print(f"Current episode length: {self.current_episode_length}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Average Episode Length: {avg_length:.1f}")
                if self.episode_rewards:
                    print(
                        f"Recent episodes: {list(self.episode_rewards)[-5:]}")
                    print(
                        f"Recent episode lengths: {list(self.episode_lengths)[-5:]}")
                if losses:
                    print(f"Policy Loss: {losses['policy_loss']:.4f}")
                    print(f"Value Loss: {losses['value_loss']:.4f}")
                print("-" * 50)

        print("Training completed!")

    def plot_results(self, save_path=None):
        """Plot training results"""
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"ppo_nes_training_results_{timestamp}.png"

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: PPO Training Rewards
        if self.training_rewards:
            ax1.plot(self.training_rewards,
                     label='PPO Training Rewards', color='blue', alpha=0.7)
            ax1.set_title('PPO Training Rewards Over Time')
            ax1.set_xlabel('PPO Updates')
            ax1.set_ylabel('Average Reward')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

        # Plot 2: NES Evolution Rewards
        if self.nes_rewards:
            ax2.plot(self.nes_rewards, label='NES Best Fitness',
                     color='red', marker='o')
            ax2.set_title('NES Evolution Progress')
            ax2.set_xlabel('Generations')
            ax2.set_ylabel('Best Fitness')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        # Plot 3: Episode Rewards Distribution
        if self.episode_rewards:
            ax3.hist(list(self.episode_rewards),
                     bins=20, alpha=0.7, color='green')
            ax3.set_title('Episode Rewards Distribution')
            ax3.set_xlabel('Reward')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)

        # Plot 4: Episode Lengths Distribution
        if self.episode_lengths:
            ax4.hist(list(self.episode_lengths),
                     bins=20, alpha=0.7, color='orange')
            ax4.set_title('Episode Lengths Distribution')
            ax4.set_xlabel('Episode Length')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Results plot saved to: {save_path}")

    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'nes_mu': self.nes.mu,
            'nes_best_params': self.nes.best_params,
            'nes_best_fitness': self.nes.best_fitness,
            'training_rewards': self.training_rewards,
            'nes_rewards': self.nes_rewards,
            'generation': self.generation
        }, filepath)
        print(f"Model saved to: {filepath}")


def main():
    """Main training function"""
    # Create agent
    agent = PPONESAgent(
        env_name="ALE/Assault-v5",
        lr=2.5e-4,
        gamma=0.99,
        clip_ratio=0.1,
        update_epochs=4,
        batch_size=32,
        buffer_size=128,
        nes_population_size=15,
        nes_sigma=0.02,
        nes_frequency=8
    )

    # Train the agent
    agent.train(total_timesteps=50000, log_interval=5)

    # Plot results
    agent.plot_results()

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"ppo_nes_assault_model_{timestamp}.pth"
    agent.save_model(model_path)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
