import os
import time
import ale_py # This is the Atari environment and should be imported to avoid errors.
import random
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from collections import deque
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation as FrameStack

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        """
        Get the output size of the convolutional layers
        Args:
            shape: The shape of the input tensor (4, 84, 84)
        Returns:
            The output size of the convolutional layers (512)
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class DQNAgent:
    def __init__(self, state_shape, n_actions, device):
        self.device = device
        self.n_actions = n_actions
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        self.memory = deque(maxlen=10000)
        
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filepath):
        """
        Save the trained model weights
        Args:
            filepath: Path where to save the model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load pre-trained model weights
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")

def plot_curves(rewards_history, losses_history):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('DQN Training Rewards for AirRaid')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(losses_history)
    plt.title('DQN Training Losses')
    plt.xlabel('Episode') 
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('dqn_training.png')
    plt.close()

def train(args):
    env = gym.make('ALE/AirRaid-v5')
    env = AtariPreprocessing(env, frame_skip=1, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, stack_size=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    state_shape = (4, 84, 84)
    n_actions = env.action_space.n
    
    agent = DQNAgent(state_shape, n_actions, device)
    rewards = []
    losses = []
    episode_times = []
    episodes = args.episodes
    
    print(f"Starting DQN training for {episodes} episodes...")
    print(f"State shape: {state_shape}, Actions: {n_actions}")
    print("-" * 80)
    
    # Create dqn directory if it doesn't exist
    os.makedirs("dqn", exist_ok=True)
    
    # Generate timestamp for this training session
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training session ID: {session_timestamp}")
    print("-" * 80)
    
    training_start_time = time.time()
    
    for episode in range(episodes):
        episode_start_time = time.time()
        
        state, _ = env.reset()
        total_reward = 0
        done = False
        episode_losses = []
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.memorize(state, action, reward, next_state, done)
            loss = agent.train()
            
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                agent.update_target_network()
                rewards.append(total_reward)
                
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                losses.append(avg_loss)
                
                episode_end_time = time.time()
                episode_duration = episode_end_time - episode_start_time
                episode_times.append(episode_duration)
                
                avg_episode_time = np.mean(episode_times)
                total_elapsed = episode_end_time - training_start_time
                
                if episode > 0:
                    remaining_episodes = episodes - (episode + 1)
                    estimated_remaining = remaining_episodes * avg_episode_time
                    eta_hours = estimated_remaining / 3600
                    eta_minutes = (estimated_remaining % 3600) / 60
                else:
                    eta_hours = eta_minutes = 0
                
                window_size = min(10, len(rewards))
                avg_reward = np.mean(rewards[-window_size:])
                
                progress = (episode + 1) / episodes * 10
                memory_usage = len(agent.memory)
                
                if episode % 10 == 0 or episode < 10:
                    print(f"Episode {episode+1:4d}/{episodes} ({progress:5.1f}%) | "
                          f"Reward: {total_reward:6.1f} | "
                          f"Avg Reward (last {window_size}): {avg_reward:6.1f} | "
                          f"Loss: {avg_loss:6.4f} | "
                          f"Epsilon: {agent.epsilon:.3f} | "
                          f"Memory: {memory_usage:5d}/{agent.memory.maxlen} | "
                          f"Time: {episode_duration:5.1f}s | "
                          f"ETA: {eta_hours:02.0f}:{eta_minutes:02.0f}")
                
                if (episode + 1) % 10 == 0:
                    recent_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                    avg_time_per_episode = np.mean(episode_times[-10:]) if len(episode_times) >= 10 else np.mean(episode_times)
                    
                    print(f"{'='*80}")
                    print(f"Episode {episode+1} Summary:")
                    print(f"  Recent 100 episodes avg reward: {recent_avg:.2f}")
                    print(f"  Best reward so far: {max(rewards):.2f}")
                    print(f"  Current epsilon: {agent.epsilon:.4f}")
                    print(f"  Memory buffer usage: {len(agent.memory)}/{agent.memory.maxlen}")
                    print(f"  Average time per episode (last 10): {avg_time_per_episode:.2f}s")
                    print(f"  Total training time: {total_elapsed/60:.1f} minutes")
                    if episode < episodes - 1:
                        print(f"  Estimated time remaining: {eta_hours:.0f}h {eta_minutes:.0f}m")
                    print(f"{'='*80}")
                    
                    checkpoint_path = f"dqn/dqn_checkpoint_episode_{episode+1}_{session_timestamp}.pth"
                    agent.save_model(checkpoint_path)
    
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    
    env.close()
    
    final_model_path = f"dqn/dqn_final_model_{session_timestamp}.pth"
    agent.save_model(final_model_path)
    
    print(f"\nTraining completed!")
    print(f"Final average reward (last 10 episodes): {np.mean(rewards[-10:]):.2f}")
    print(f"Best episode reward: {max(rewards):.2f}")
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
    print(f"Average time per episode: {np.mean(episode_times):.2f} seconds")
    print(f"Fastest episode: {min(episode_times):.2f} seconds")
    print(f"Slowest episode: {max(episode_times):.2f} seconds")
    
    return rewards, losses

def simulate(args):
    """
    Simulate/test a trained DQN model
    Args:
        args: Command line arguments containing model path and simulation settings
    """

    if args.render:
        env = gym.make('ALE/AirRaid-v5', render_mode='human')
    else:
        env = gym.make('ALE/AirRaid-v5')
    
    env = AtariPreprocessing(env, frame_skip=1, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, stack_size=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    state_shape = (4, 84, 84)
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions, device)
    
    model_path = args.model_path
    if model_path == "auto":
        model_path = find_latest_model()
        if model_path is None:
            print("Error: No trained models found in the 'dqn' folder!")
            print("Please train a model first or specify a valid model path.")
            return
        print(f"Automatically selected model: {model_path}")
    
    try:
        agent.load_model(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train a model first or specify a valid model path.")
        return
    
    # Set epsilon to 0 for pure exploitation (no exploration)
    original_epsilon = agent.epsilon
    # Small epsilon for occasional exploration
    agent.epsilon = 0.05  

    print(f"Starting simulation with trained model...")
    print(f"Model path: {model_path}")
    print(f"Simulation episodes: {args.sim_episodes}")
    print(f"Epsilon set to: {agent.epsilon} (was {original_epsilon})")
    print("-" * 80)
    
    simulation_rewards = []
    
    for episode in range(args.sim_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        print(f"Starting episode {episode + 1}...")
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps > 10000:  # Prevent infinite episodes
                break
        
        simulation_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.1f}, Steps = {steps}")
    
    env.close()
    
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print(f"Episodes simulated: {args.sim_episodes}")
    print(f"Average reward: {np.mean(simulation_rewards):.2f}")
    print(f"Best episode: {max(simulation_rewards):.2f}")
    print(f"Worst episode: {min(simulation_rewards):.2f}")
    print(f"Standard deviation: {np.std(simulation_rewards):.2f}")
    print("="*80)
    
    return simulation_rewards

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_min", type=float, default=0.1)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--simulate", action="store_true", help="Run simulation with trained model")
    parser.add_argument("--model_path", type=str, default="auto", help="Path to trained model for simulation (use 'auto' for latest)")
    parser.add_argument("--sim_episodes", type=int, default=5, help="Number of episodes to simulate")
    parser.add_argument("--render", action="store_true", help="Render the game during simulation")
    return parser.parse_args()

def find_latest_model():
    """
    Find the most recent final model in the dqn folder
    Returns:
        Path to the latest model or None if no models found
    """
    dqn_dir = "dqn"
    if not os.path.exists(dqn_dir):
        return None
    
    final_models = [f for f in os.listdir(dqn_dir) if f.startswith("dqn_final_model_") and f.endswith(".pth")]
    
    if not final_models:
        checkpoint_models = [f for f in os.listdir(dqn_dir) if f.startswith("dqn_checkpoint_") and f.endswith(".pth")]
        if checkpoint_models:
            checkpoint_models.sort(key=lambda x: os.path.getmtime(os.path.join(dqn_dir, x)), reverse=True)
            return os.path.join(dqn_dir, checkpoint_models[0])
        return None
    
    final_models.sort(key=lambda x: os.path.getmtime(os.path.join(dqn_dir, x)), reverse=True)
    return os.path.join(dqn_dir, final_models[0])

if __name__ == "__main__":
    args = parse_args()
    
    if args.simulate:
        simulate(args)
    else:
        rewards, losses = train(args)
        plot_curves(rewards, losses)
