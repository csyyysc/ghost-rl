import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

class Actor(nn.Module):
    
    def __init__(self, observation_space, action_space, hidden_dim=256):
        """
        Observation Space: Box(0, 255, (96, 96, 3), uint8)
        Action Space: Box([-1. 0. 0.], 1.0, (3,), float32)
        """
        super(Actor, self).__init__()
        
        """
        (3, 32): RGB Channel, 32 Filters => reduces spatial size
            Output Size: Gauss((Input - Kernel + 2 * Padding) / Stride) + 1

        First Layer
            (3, 32) => Gauss((96 - 8 + 2 * 0) / 4) + 1 = 22

        Second Layer
            (32, 23, 23) => Gauss((23 - 4 + 2 * 0) / 2) + 1 = 10

        Third Layer
            (64, 10, 10) => Gauss((10 - 3 + 2 * 0) / 1) + 1 = 10

        Fourth Layer
            (64, 8, 8) => Gauss((8 - 3 + 2 * 0) / 1) + 1 = 8
        """
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_output_size(observation_space.shape)
        
        self.shared_layers = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_space.shape[0])
        )
        
        self.log_std_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_space.shape[0])
        )
        
        self.register_buffer('action_scale', torch.FloatTensor((action_space.high - action_space.low) / 2.0))
        self.register_buffer('action_bias', torch.FloatTensor((action_space.high + action_space.low) / 2.0))
        
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
    
    def _get_conv_output_size(self, input_shape):
        """Calculate the output size of convolutional layers
        
        Size: 1 * 8 * 8 * 64 = 4096
        """

        dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.size()))
    
    def forward(self, state):
        conv_out = self.conv_layers(state)
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        features = self.shared_layers(conv_out)
        
        mean = self.mean_net(features)
        log_std = self.log_std_net(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mean, log_std
    
    def sample_action(self, state):
        """Sample action with reparameterization trick"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


class Critic(nn.Module):
    
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_output_size(observation_space.shape)
        
        self.q1_net = nn.Sequential(
            nn.Linear(conv_out_size + action_space.shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_net = nn.Sequential(
            nn.Linear(conv_out_size + action_space.shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def _get_conv_output_size(self, input_shape):
        dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.size()))
    
    def forward(self, state, action):
        conv_out = self.conv_layers(state)
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        x = torch.cat([conv_out, action], dim=1)
        
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        
        return q1, q2


class ReplayBuffer:
    
    def __init__(self, capacity, device):
        self.device = device
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
        
        if state.dim() == 4 and state.size(0) == 1:
            state = state.squeeze(0)
        if next_state.dim() == 4 and next_state.size(0) == 1:
            next_state = next_state.squeeze(0)
        if action.dim() == 2 and action.size(0) == 1:
            action = action.squeeze(0)
            
        self.states[self.position] = state.cpu()
        self.actions[self.position] = action.cpu()
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state.cpu()
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch_indices = random.sample(range(len(self.states)), batch_size)
        
        states = torch.stack([self.states[i] for i in batch_indices]).to(self.device)
        actions = torch.stack([self.actions[i] for i in batch_indices]).to(self.device)
        rewards = torch.FloatTensor([self.rewards[i] for i in batch_indices]).to(self.device)
        next_states = torch.stack([self.next_states[i] for i in batch_indices]).to(self.device)
        dones = torch.FloatTensor([self.dones[i] for i in batch_indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.states)


class SACAgent:
    
    def __init__(self, env, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, 
                 batch_size=256, buffer_size=1000000, auto_entropy_tuning=True):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning
        
        self.actor = Actor(env.observation_space, env.action_space).to(self.device)
        self.critic = Critic(env.observation_space, env.action_space).to(self.device)
        self.critic_target = Critic(env.observation_space, env.action_space).to(self.device)
        self.replay_buffer = ReplayBuffer(buffer_size, self.device)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        if auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha
        
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        

        self.training_metrics = {
            'q1_values': deque(maxlen=1000),
            'q2_values': deque(maxlen=1000),
            'entropy': deque(maxlen=1000),
            'policy_loss': deque(maxlen=1000),
            'value_loss': deque(maxlen=1000),
            'alpha_values': deque(maxlen=1000),
            'episode_count': 0
        }
    
    @property
    def alpha(self):
        if self.auto_entropy_tuning:
            return self.log_alpha.exp()
        else:
            return self._alpha
    
    @alpha.setter 
    def alpha(self, value):
        self._alpha = value
    
    def preprocess_state(self, state):
        if isinstance(state, tuple):
            state = state[0]
        
        state = np.transpose(state, (2, 0, 1))
        state = state / 255.0
        return torch.FloatTensor(state).to(self.device)
    
    def select_action(self, state, evaluate=False):
        if state.dim() == 3:
            state = state.unsqueeze(0)
            
        try:
            if evaluate:
                with torch.no_grad():
                    _, _, action = self.actor.sample_action(state)
            else:
                with torch.no_grad():
                    action, _, _ = self.actor.sample_action(state)
            
            return action
        except Exception as e:
            print(f"❌ Error in select_action: {e}")
            print(f"State shape: {state.shape}, device: {state.device}")
            raise e
    
    def collect_experience(self, num_steps):
        state, _ = self.env.reset()
        state = self.preprocess_state(state)
        
        episode_reward = 0
        episode_length = 0
        
        for _ in range(num_steps):
            action = self.select_action(state)
            if action.dim() == 2 and action.size(0) == 1:
                action_for_env = action.squeeze(0)
            else:
                action_for_env = action
            
            if action.dim() == 1:
                action_for_env = action.cpu().numpy()
            else:
                action_for_env = action.squeeze(0).cpu().numpy()
            next_state, reward, done, truncated, _ = self.env.step(action_for_env)
            next_state = self.preprocess_state(next_state)
            
            self.replay_buffer.push(state, action, reward, next_state, done or truncated)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if done or truncated:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                state, _ = self.env.reset()
                state = self.preprocess_state(state)
                episode_reward = 0
                episode_length = 0
    
    def update_policy(self):
        """Update SAC policy networks"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample_action(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        

        self.training_metrics['q1_values'].extend(current_q1.detach().cpu().numpy().flatten())
        self.training_metrics['q2_values'].extend(current_q2.detach().cpu().numpy().flatten())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        new_actions, log_probs, _ = self.actor.sample_action(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        

        entropy = -log_probs.mean()
        self.training_metrics['entropy'].append(entropy.item())
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        

        self.training_metrics['policy_loss'].append(actor_loss.item())
        self.training_metrics['value_loss'].append(critic_loss.item())
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = None
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        current_alpha = self.alpha.item() if hasattr(self.alpha, 'item') else self.alpha
        self.training_metrics['alpha_values'].append(current_alpha)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else 0,
            'alpha': current_alpha,
            'entropy': entropy.item(),
            'mean_q1': current_q1.mean().item(),
            'mean_q2': current_q2.mean().item(),
            'std_q1': current_q1.std().item(),
            'std_q2': current_q2.std().item()
        }
    
    def save_model(self, filepath):

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
        }, filepath)
    
    def load_model(self, filepath):

        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha.data.copy_(checkpoint['log_alpha'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

    def get_training_summary(self):

        summary = {
            'total_episodes': self.training_metrics['episode_count'],
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if len(self.episode_rewards) > 1 else 0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'buffer_size': len(self.replay_buffer),
            'recent_performance': {
                'avg_q1': np.mean(list(self.training_metrics['q1_values'])[-1000:]) if self.training_metrics['q1_values'] else 0,
                'avg_q2': np.mean(list(self.training_metrics['q2_values'])[-1000:]) if self.training_metrics['q2_values'] else 0,
                'avg_entropy': np.mean(list(self.training_metrics['entropy'])[-1000:]) if self.training_metrics['entropy'] else 0,
                'avg_alpha': np.mean(list(self.training_metrics['alpha_values'])[-1000:]) if self.training_metrics['alpha_values'] else 0,
                'avg_policy_loss': np.mean(list(self.training_metrics['policy_loss'])[-1000:]) if self.training_metrics['policy_loss'] else 0,
                'avg_value_loss': np.mean(list(self.training_metrics['value_loss'])[-1000:]) if self.training_metrics['value_loss'] else 0,
            }
        }
        return summary
    
    def print_training_summary(self):

        summary = self.get_training_summary()
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Average Reward: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"Average Episode Length: {summary['avg_episode_length']:.1f}")
        print(f"Replay Buffer Size: {summary['buffer_size']}")
        print(f"\nRecent Performance Metrics (last 1000 updates):")
        print(f"  Average Q1 Value: {summary['recent_performance']['avg_q1']:.2f}")
        print(f"  Average Q2 Value: {summary['recent_performance']['avg_q2']:.2f}")
        print(f"  Average Entropy: {summary['recent_performance']['avg_entropy']:.4f}")
        print(f"  Average Alpha: {summary['recent_performance']['avg_alpha']:.4f}")
        print(f"  Average Policy Loss: {summary['recent_performance']['avg_policy_loss']:.4f}")
        print(f"  Average Value Loss: {summary['recent_performance']['avg_value_loss']:.4f}")
        print(f"{'='*60}\n")


def train():
    """Train SAC agent on CarRacing-v3 environment"""

    env = gym.make('CarRacing-v3', continuous=True, render_mode=None)
    agent = SACAgent(env)
    
    total_timesteps = 100000
    update_frequency = 1  # Update after every step for SAC
    warmup_steps = 10000  # Random actions for initial exploration
    
    print(f"Training SAC on CarRacing-v3")
    print(f"Device: {agent.device}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Warmup steps: {warmup_steps}")
    print("-" * 50)
    
    timestep = 0
    update_count = 0
    
    losses = []
    rewards = []
    
    state, _ = env.reset()
    state = agent.preprocess_state(state)
    episode_reward = 0
    episode_length = 0
    
    while timestep < total_timesteps:
        if timestep < warmup_steps:
            action = torch.FloatTensor(env.action_space.sample()).to(agent.device)
        else:
            if timestep == warmup_steps:
                print(f"🚀 Starting policy-based training at timestep {timestep}")
            action = agent.select_action(state)
            if action.dim() == 2 and action.size(0) == 1:
                action = action.squeeze(0)
        
        if action.dim() == 1:
            action_for_env = action.cpu().numpy()
        else:
            action_for_env = action.squeeze(0).cpu().numpy()
        next_state, reward, done, truncated, _ = env.step(action_for_env)
        next_state = agent.preprocess_state(next_state)
        
        agent.replay_buffer.push(state, action, reward, next_state, done or truncated)
        
        episode_reward += reward
        episode_length += 1
        timestep += 1
        
        state = next_state
        
        if timestep > warmup_steps and timestep % update_frequency == 0:
            loss_dict = agent.update_policy()
            if loss_dict:
                losses.append(loss_dict)
                update_count += 1
        
        if done or truncated:
            agent.episode_rewards.append(episode_reward)
            agent.episode_lengths.append(episode_length)
            agent.training_metrics['episode_count'] += 1
            
            # Log episode information
            print(f"Episode {agent.training_metrics['episode_count']}: "
                  f"Reward = {episode_reward:.2f}, Length = {episode_length}, "
                  f"Timestep = {timestep}")
            
            state, _ = env.reset()
            state = agent.preprocess_state(state)
            episode_reward = 0
            episode_length = 0
        
        if timestep % 10000 == 0 and len(agent.episode_rewards) > 0:
            avg_reward = np.mean(agent.episode_rewards)
            avg_length = np.mean(agent.episode_lengths)
            rewards.append(avg_reward)
            
            # Calculate detailed training statistics
            stats = {
                'timestep': timestep,
                'episodes': agent.training_metrics['episode_count'],
                'avg_reward': avg_reward,
                'std_reward': np.std(agent.episode_rewards),
                'avg_length': avg_length,
                'buffer_size': len(agent.replay_buffer)
            }
            
            if losses and agent.training_metrics['entropy']:
                latest_loss = losses[-1]
                
                # Get recent training metrics
                recent_q1 = list(agent.training_metrics['q1_values'])[-100:] if agent.training_metrics['q1_values'] else [0]
                recent_q2 = list(agent.training_metrics['q2_values'])[-100:] if agent.training_metrics['q2_values'] else [0]
                recent_entropy = list(agent.training_metrics['entropy'])[-100:] if agent.training_metrics['entropy'] else [0]
                recent_alpha = list(agent.training_metrics['alpha_values'])[-100:] if agent.training_metrics['alpha_values'] else [0]
                
                stats.update({
                    'critic_loss': latest_loss['critic_loss'],
                    'actor_loss': latest_loss['actor_loss'],
                    'alpha_loss': latest_loss['alpha_loss'],
                    'alpha': latest_loss['alpha'],
                    'entropy': latest_loss['entropy'],
                    'mean_q1': latest_loss['mean_q1'],
                    'mean_q2': latest_loss['mean_q2'],
                    'avg_q1_recent': np.mean(recent_q1),
                    'avg_q2_recent': np.mean(recent_q2),
                    'avg_entropy_recent': np.mean(recent_entropy),
                    'avg_alpha_recent': np.mean(recent_alpha)
                })
                
                print(f"\n{'='*80}")
                print(f"TRAINING EPOCH SUMMARY - Timestep {timestep}")
                print(f"{'='*80}")
                print(f"Episodes Completed: {stats['episodes']}")
                print(f"Average Reward (last 100): {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
                print(f"Average Episode Length: {stats['avg_length']:.1f}")
                print(f"Replay Buffer Size: {stats['buffer_size']}")
                print(f"\nTraining Metrics:")
                print(f"  Critic Loss: {stats['critic_loss']:.4f}")
                print(f"  Actor Loss: {stats['actor_loss']:.4f}")
                print(f"  Alpha Loss: {stats['alpha_loss']:.4f}")
                print(f"  Current Alpha: {stats['alpha']:.4f}")
                print(f"  Current Entropy: {stats['entropy']:.4f}")
                print(f"\nQ-Value Statistics:")
                print(f"  Mean Q1: {stats['mean_q1']:.2f} | Mean Q2: {stats['mean_q2']:.2f}")
                print(f"  Avg Q1 (recent): {stats['avg_q1_recent']:.2f}")
                print(f"  Avg Q2 (recent): {stats['avg_q2_recent']:.2f}")
                print(f"\nExploration Statistics:")
                print(f"  Average Entropy (recent): {stats['avg_entropy_recent']:.4f}")
                print(f"  Average Alpha (recent): {stats['avg_alpha_recent']:.4f}")
                print(f"{'='*80}\n")
            else:
                print(f"\n{'='*60}")
                print(f"WARMUP PHASE - Timestep {timestep}")
                print(f"{'='*60}")
                print(f"Episodes Completed: {stats['episodes']}")
                print(f"Average Reward (last 100): {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
                print(f"Average Episode Length: {stats['avg_length']:.1f}")
                print(f"Replay Buffer Size: {stats['buffer_size']}")
                print(f"{'='*60}\n")
        
        if timestep % 100000 == 0 and timestep > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            agent.save_model(f"sac_model_{timestamp}.pt")
            
            # Print comprehensive training summary
            print(f"\n🔄 MODEL CHECKPOINT SAVED AT TIMESTEP {timestep}")
            agent.print_training_summary()
        
        # Additional periodic detailed summary
        if timestep % 50000 == 0 and timestep > 0:
            print(f"\n📊 DETAILED TRAINING ANALYSIS - Timestep {timestep}")
            agent.print_training_summary()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.save_model(f"sac_model_final_{timestamp}.pt")
    
    if rewards:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(rewards)
        plt.title('Average Episode Reward')
        plt.xlabel('Updates (x10k timesteps)')
        plt.ylabel('Reward')
        
        if losses:
            plt.subplot(1, 3, 2)
            critic_losses = [loss['critic_loss'] for loss in losses]
            actor_losses = [loss['actor_loss'] for loss in losses]
            plt.plot(critic_losses, label='Critic Loss')
            plt.plot(actor_losses, label='Actor Loss')
            plt.title('Training Losses')
            plt.xlabel('Update')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            alphas = [loss['alpha'] for loss in losses]
            plt.plot(alphas)
            plt.title('Temperature (Alpha)')
            plt.xlabel('Update')
            plt.ylabel('Alpha')
        
        plt.tight_layout()
        plt.savefig(f'sac_training_progress_{timestamp}.png')
        plt.show()
    
    env.close()
    return agent


def simulate(model_path):
    """Simulate trained SAC agent"""
    env = gym.make('CarRacing-v3', continuous=True, render_mode='human')
    agent = SACAgent(env)
    agent.load_model(model_path)
    
    num_episodes = 5
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = agent.preprocess_state(state)
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            
            # Handle action dimensions for environment step
            if action.dim() == 2 and action.size(0) == 1:
                action_for_env = action.squeeze(0).cpu().numpy()
            else:
                action_for_env = action.cpu().numpy()
                
            next_state, reward, done, truncated, _ = env.step(action_for_env)
            state = agent.preprocess_state(next_state)
            
            episode_reward += reward
            done = done or truncated
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()


if __name__ == "__main__":
    print("CarRacing-v3 (SAC) training...")
    agent = train()
    
    # Uncomment to test a trained model
    # simulate("sac_model_final_20250614_202801.pt")
