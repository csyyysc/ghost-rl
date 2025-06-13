import os
import json
import neat
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.distributions import Normal

class PPONetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, genome=None, architecture=None):
        super(PPONetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if genome is not None:
            # Initialize with NEAT genome network architecture
            self.actor = self._create_network_from_genome(genome, is_actor=True)
            self.critic = self._create_network_from_genome(genome, is_actor=False)
        elif architecture is not None:
            # Initialize with specific architecture
            self.actor = self._create_network_from_architecture(architecture['actor'], is_actor=True)
            self.critic = self._create_network_from_architecture(architecture['critic'], is_actor=False)
        else:
            # Default network architecture if no genome 
            self.actor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
            self.critic = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        self.log_std = nn.Parameter(torch.zeros(output_dim))
    
    def _create_network_from_architecture(self, architecture, is_actor=True):
        """Create a PyTorch network from a specific architecture"""
        layers = []
        for i in range(len(architecture) - 1):
            in_features = architecture[i][1] if i == 0 else architecture[i][0]
            out_features = architecture[i+1][0]
            layers.append(nn.Linear(in_features, out_features))
            if i < len(architecture) - 2:  # Don't add ReLU after last layer
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def _create_network_from_genome(self, genome, is_actor=True):
        """Create a PyTorch network from NEAT genome efficiently"""
        nodes = sorted(genome.nodes.keys())
        input_size = self.input_dim
        output_size = self.output_dim if is_actor else 1
        
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        hidden_nodes = [n for n in nodes if input_size <= n < input_size + self.hidden_dim]
        
        if not hidden_nodes:
            output_layer = nn.Linear(input_size, output_size)
            for i in range(output_size):
                for conn in genome.connections.values():
                    if conn.enabled and conn.key[1] >= input_size + self.hidden_dim:
                        if conn.key[0] < input_size:
                            output_layer.weight.data[i, conn.key[0]] = conn.weight
            return nn.Sequential(output_layer)
        
        layers = []
        current_size = len(hidden_nodes)
        
        first_hidden = nn.Linear(input_size, current_size)
        for conn in genome.connections.values():
            if conn.enabled and conn.key[1] in hidden_nodes:
                if conn.key[0] < input_size:
                    first_hidden.weight.data[hidden_nodes.index(conn.key[1]), conn.key[0]] = conn.weight
        layers.append(first_hidden)
        layers.append(nn.ReLU())
        
        if len(hidden_nodes) > 1:
            hidden_layer = nn.Linear(current_size, current_size)
            for conn in genome.connections.values():
                if conn.enabled and conn.key[1] in hidden_nodes:
                    if conn.key[0] in hidden_nodes:
                        i = hidden_nodes.index(conn.key[1])
                        j = hidden_nodes.index(conn.key[0])
                        hidden_layer.weight.data[i, j] = conn.weight
            layers.append(hidden_layer)
            layers.append(nn.ReLU())
        
        output_layer = nn.Linear(current_size, output_size)
        # Fix output node index calculation
        output_nodes = [n for n in nodes if n >= input_size + self.hidden_dim]
        for conn in genome.connections.values():
            if conn.enabled and conn.key[1] in output_nodes:
                if conn.key[0] in hidden_nodes:
                    i = output_nodes.index(conn.key[1]) % output_size  # Ensure index is within bounds
                    j = hidden_nodes.index(conn.key[0])
                    output_layer.weight.data[i, j] = conn.weight
        
        layers.append(output_layer)
        return nn.Sequential(*layers)
        
    def forward(self, x):
        return self.actor(x), self.critic(x)
    
    def get_action(self, state):
        mean, _ = self.forward(state)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        return action, dist.log_prob(action)

class NEATWrapper:
    def __init__(self, config_path):
        self.config = neat.Config(
            neat.DefaultGenome, 
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet, 
            neat.DefaultStagnation,
            config_path)
        self.population = neat.Population(self.config)
        
    def get_action(self, genome, net, state):
        output = net.activate(state)
        return output[0]  # Mountain Car has 1D action space
    
    def save_genome(self, genome, filename):
        """Save NEAT genome to a file"""
        genome_data = {
            'nodes': {str(k): {
                'bias': v.bias,
                'response': v.response,
                'activation': v.activation,
                'aggregation': v.aggregation
            } for k, v in genome.nodes.items()},
            'connections': {str(k): {
                'weight': v.weight,
                'enabled': v.enabled
            } for k, v in genome.connections.items()},
            'fitness': float(genome.fitness),
            'key': int(genome.key)
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(genome_data, f, indent=4)
    
    def load_genome(self, filename):
        """Load NEAT genome from a file"""
        with open(filename, 'r', encoding='utf-8') as f:
            genome_data = json.load(f)
        
        genome = neat.DefaultGenome(genome_data['key'])
        genome.fitness = genome_data['fitness']
        
        for node_key, node_data in genome_data['nodes'].items():
            node = neat.genome.DefaultNodeGene(int(node_key))
            node.bias = node_data['bias']
            node.response = node_data['response']
            node.activation = node_data['activation']
            node.aggregation = node_data['aggregation']
            genome.nodes[int(node_key)] = node
        
        for conn_key, conn_data in genome_data['connections'].items():
            conn = neat.genome.DefaultConnectionGene(int(conn_key))
            conn.weight = conn_data['weight']
            conn.enabled = conn_data['enabled']
            genome.connections[int(conn_key)] = conn
        
        return genome

class PPO_NEAT:
    def __init__(self, env_name, hidden_dim=64, lr=3e-4, gamma=0.99, epsilon=0.2, 
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, render_mode=None,
                 lambda_gae=0.95):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.shape[0]
        self.hidden_dim = hidden_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.neat_wrapper = NEATWrapper('neat_config.txt')
        self.ppo_network = PPONetwork(self.input_dim, hidden_dim, self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.ppo_network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.lambda_gae = lambda_gae
        
        self.current_genome = None
        self.current_genome_fitness = float('-inf')

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            if done:
                gae = 0
            delta = r + self.gamma * next_value * (1 - done) - v
            gae = delta + self.gamma * self.lambda_gae * (1 - done) * gae
            advantages.insert(0, gae)
            next_value = v
        return torch.tensor(advantages, device=self.device)
    
    def update(self, states, actions, old_log_probs, rewards, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        with torch.no_grad():
            _, values = self.ppo_network(states)
            next_value = values[-1]
            advantages = self.compute_gae(rewards, values.cpu().numpy(), next_value.cpu().numpy(), dones)
            returns = advantages + values.detach()
        
        # PPO update with more iterations for non-ES training
        for i in range(10):
            new_actions, new_log_probs = self.ppo_network.get_action(states)
            _, new_values = self.ppo_network(states)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - new_values).pow(2).mean()
            entropy_loss = -0.01 * new_log_probs.mean()
            
            total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ppo_network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Print loss components for debugging
            if i  == 9:  # Only print for first iteration
                print(f"  Loss components:")
                print(f"    Actor loss: {actor_loss.item():.4f}")
                print(f"    Critic loss: {critic_loss.item():.4f}")
                print(f"    Entropy loss: {entropy_loss.item():.4f}")
                print(f"    Total loss: {total_loss.item():.4f}")
    
    def evaluate_neat(self, genomes, config):
        """Evaluate NEAT genomes for fitness"""
        for _, genome in genomes:
            # Create new PPO network with this genome
            network = PPONetwork(self.input_dim, self.hidden_dim, self.output_dim, 
                               genome=genome).to(self.device)
            
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).to(self.device)
                action, _ = network.get_action(state_tensor)
                action = action.cpu().numpy()
                
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                
            # Ensure minimum fitness to prevent extinction
            genome.fitness = max(total_reward, 0.1)
            
        return None

    def print_network_structure(self, genome):
        """Print the structure of the NEAT network"""
        print("\nNetwork Structure:")
        print("-----------------")
        
        # Get node information
        input_nodes = [n for n in genome.nodes.keys() if n < self.input_dim]
        hidden_nodes = [n for n in genome.nodes.keys() if self.input_dim <= n < self.input_dim + self.hidden_dim]
        output_nodes = [n for n in genome.nodes.keys() if n >= self.input_dim + self.hidden_dim]
        
        print(f"Input nodes: {input_nodes}")
        print(f"Hidden nodes: {hidden_nodes}")
        print(f"Output nodes: {output_nodes}")
        
        # Count enabled connections
        enabled_connections = sum(1 for conn in genome.connections.values() if conn.enabled)
        total_connections = len(genome.connections)
        
        print(f"\nConnections:")
        print(f"  Enabled: {enabled_connections}")
        print(f"  Total: {total_connections}")
        print(f"  Enabled ratio: {enabled_connections/total_connections:.2%}")
        
        # Print connection details
        print("\nConnection Details:")
        for conn_key, conn in genome.connections.items():
            if conn.enabled:
                print(f"  {conn_key[0]} -> {conn_key[1]} (weight: {conn.weight:.4f})")
        
        print("-----------------\n")

    def train_without_es(self, num_episodes=1000):
        """Train the agent using standard PPO without evolutionary strategies"""
        episode_rewards = []
        best_reward = float('-inf')
        no_improvement_count = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            states, actions, rewards, log_probs, dones = [], [], [], [], []
            
            while True:
                state_tensor = torch.FloatTensor(state).to(self.device)
                action, log_prob = self.ppo_network.get_action(state_tensor)
                action = action.cpu().numpy()
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob.cpu().detach().numpy())
                dones.append(done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            self.update(states, actions, log_probs, rewards, dones)
            episode_rewards.append(episode_reward)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            print(f"Episode {episode}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Best Reward: {best_reward:.2f}")
            print(f"  Episodes without improvement: {no_improvement_count}")
            print(f"  Trajectory length: {len(states)}")
            print(f"  Mean reward per step: {episode_reward/len(states):.2f}")
            
            if no_improvement_count >= 100:
                print("Performance plateaued, stopping training")
                break
            
        return episode_rewards

    def train(self, num_episodes=100, use_es=True):
        """Train the agent using either PPO with NEAT or standard PPO"""
        if use_es:
            return self.train_with_es(num_episodes)
        else:
            return self.train_without_es(num_episodes)

    def train_with_es(self, num_episodes=100):
        """Train the agent using PPO with NEAT evolutionary strategies"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            try:
                self.neat_wrapper.population.run(self.evaluate_neat, 1)
                # Get new best genome
                new_genome = self.neat_wrapper.population.best_genome
                if new_genome is not None:
                    new_fitness = new_genome.fitness
                    # Update if we have no genome yet or if new genome is better
                    if (self.current_genome is None or 
                        (new_fitness > self.current_genome_fitness)):
                        self.current_genome = new_genome
                        self.current_genome_fitness = new_fitness
                        # Create new PPO network with evolved genome
                        self.ppo_network = PPONetwork(self.input_dim, self.hidden_dim, 
                                                    self.output_dim, genome=self.current_genome).to(self.device)
                        self.optimizer = optim.Adam(self.ppo_network.parameters(), lr=3e-4)
                        print(f"New best genome found! Fitness: {new_fitness:.2f}")
                        # Print network structure for new best genome
                        self.print_network_structure(self.current_genome)
            except neat.population.CompleteExtinctionException:
                print("Population went extinct, reinitializing...")
                self.neat_wrapper.population = neat.Population(self.neat_wrapper.config)
                self.neat_wrapper.population.run(self.evaluate_neat, 1)
            
            state, _ = self.env.reset()
            episode_reward = 0
            states, actions, rewards, log_probs, dones = [], [], [], [], []
            
            while True:
                state_tensor = torch.FloatTensor(state).to(self.device)
                action, log_prob = self.ppo_network.get_action(state_tensor)
                action = action.cpu().numpy()
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob.cpu().detach().numpy())
                dones.append(done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            self.update(states, actions, log_probs, rewards, dones)
            
            episode_rewards.append(episode_reward)
            genome_fitness = self.current_genome_fitness if self.current_genome is not None else float('-inf')
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Genome Fitness: {genome_fitness:.2f}")
            
        return episode_rewards
    
    def save_model(self, path='.'):
        """Save both PPO and NEAT models in the current directory"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = f'model_{timestamp}'
            os.makedirs(model_dir, exist_ok=True)
            
            # Save PPO model
            ppo_path = os.path.join(model_dir, 'ppo_model.pt')
            
            # Get network architecture information
            actor_layers = []
            critic_layers = []
            for name, param in self.ppo_network.named_parameters():
                if 'weight' in name:
                    if 'actor' in name:
                        actor_layers.append(param.shape)
                    elif 'critic' in name:
                        critic_layers.append(param.shape)
            
            ppo_data = {
                'model_state_dict': self.ppo_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'hidden_dim': self.hidden_dim,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'network_type': 'neat' if self.current_genome is not None else 'default',
                'actor_architecture': actor_layers,
                'critic_architecture': critic_layers
            }
            torch.save(ppo_data, ppo_path)
            
            # Verify PPO model was saved correctly
            if not os.path.exists(ppo_path):
                raise IOError(f"Failed to save PPO model to {ppo_path}")
            
            # Save NEAT genome if available
            genome = self.current_genome
            neat_path = None
            if genome is not None:
                neat_path = os.path.join(model_dir, 'neat_genome.json')
                try:
                    self.neat_wrapper.save_genome(genome, neat_path)
                    # Verify NEAT genome was saved correctly
                    if not os.path.exists(neat_path):
                        raise IOError(f"Failed to save NEAT genome to {neat_path}")
                except Exception as e:
                    print(f"Warning: Failed to save NEAT genome: {e}")
                    neat_path = None
            
            # Save configuration
            config = {
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'ppo_path': 'ppo_model.pt',  # Relative paths
                'neat_path': 'neat_genome.json' if neat_path else None,
                'timestamp': timestamp,
                'env_name': self.env.spec.id if self.env.spec else 'MountainCarContinuous-v0',
                'network_type': 'neat' if genome is not None else 'default',
                'actor_architecture': actor_layers,
                'critic_architecture': critic_layers
            }
            config_path = os.path.join(model_dir, 'config.json')
            
            # Save and verify config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            if not os.path.exists(config_path):
                raise IOError(f"Failed to save config to {config_path}")
            
            print(f"\nModel saved successfully:")
            print(f"Directory: {model_dir}")
            print(f"PPO model: {ppo_path}")
            if neat_path:
                print(f"NEAT genome: {neat_path}")
            print(f"Config file: {config_path}")
            
            return config_path
            
        except Exception as e:
            print(f"Error saving model: {e}")
            # Clean up any partially saved files
            if 'model_dir' in locals() and os.path.exists(model_dir):
                import shutil
                shutil.rmtree(model_dir)
            raise
    
    @classmethod
    def load_model(cls, config_path, env_name='MountainCarContinuous-v0', render_mode=None):
        """Load saved models and create a new instance"""
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            # Validate that we're loading a config file, not a model file
            if not config_path.endswith('.json'):
                raise ValueError(
                    f"Expected a JSON config file, but got: {config_path}\n"
                    "Please provide the path to the config.json file, not the model file."
                )
                
            model_dir = os.path.dirname(config_path)
            
            # Load configuration
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except UnicodeDecodeError:
                raise ValueError(
                    f"Config file appears to be corrupted or in wrong format: {config_path}\n"
                    "Please ensure you're providing the path to config.json, not ppo_model.pt"
                )
            except json.JSONDecodeError:
                raise ValueError(f"Config file is not valid JSON: {config_path}")
            
            # Create new instance with saved hyperparameters
            instance = cls(
                env_name, 
                hidden_dim=config.get('hidden_dim', 64),
                lr=config.get('lr', 3e-4),
                gamma=config.get('gamma', 0.99),
                epsilon=config.get('epsilon', 0.2),
                value_coef=config.get('value_coef', 0.5),
                entropy_coef=config.get('entropy_coef', 0.01),
                max_grad_norm=config.get('max_grad_norm', 0.5),
                render_mode=render_mode,
                lambda_gae=config.get('lambda_gae', 0.95)
            )
            
            # Load PPO model first to get architecture info
            ppo_path = os.path.join(model_dir, config['ppo_path'])
            if not os.path.exists(ppo_path):
                raise FileNotFoundError(f"PPO model file not found: {ppo_path}")
            
            checkpoint = torch.load(ppo_path, map_location=instance.device)
            network_type = checkpoint.get('network_type', 'default')
            
            # Load NEAT genome if it's a NEAT network
            genome = None
            if network_type == 'neat' and config.get('neat_path'):
                neat_path = os.path.join(model_dir, config['neat_path'])
                if os.path.exists(neat_path):
                    try:
                        genome = instance.neat_wrapper.load_genome(neat_path)
                        instance.neat_wrapper.population.best_genome = genome
                    except Exception as e:
                        print(f"Warning: Could not load NEAT genome: {e}")
            
            # Create PPO network with the correct architecture
            if genome is not None:
                instance.ppo_network = PPONetwork(
                    instance.input_dim, 
                    instance.hidden_dim, 
                    instance.output_dim,
                    genome=genome
                ).to(instance.device)
            else:
                # Create network with saved architecture
                actor_arch = checkpoint.get('actor_architecture', None)
                critic_arch = checkpoint.get('critic_architecture', None)
                
                if actor_arch and critic_arch:
                    # Create custom network with saved architecture
                    instance.ppo_network = PPONetwork(
                        instance.input_dim,
                        instance.hidden_dim,
                        instance.output_dim,
                        architecture={'actor': actor_arch, 'critic': critic_arch}
                    ).to(instance.device)
                else:
                    # If no architecture info, create default network
                    instance.ppo_network = PPONetwork(
                        instance.input_dim,
                        instance.hidden_dim,
                        instance.output_dim
                    ).to(instance.device)
            
            # Load the model weights
            try:
                # Get the state dict from the checkpoint
                state_dict = checkpoint['model_state_dict']
                
                # Filter out unexpected keys
                model_state_dict = instance.ppo_network.state_dict()
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
                
                # Load the filtered state dict
                instance.ppo_network.load_state_dict(filtered_state_dict, strict=False)
                
                # Create a new optimizer with the current model parameters
                instance.optimizer = optim.Adam(instance.ppo_network.parameters(), lr=config.get('lr', 3e-4))
                
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except Exception as e:
                        print(f"Warning: Could not load optimizer state: {e}")
                        print("Continuing with fresh optimizer state")
            except Exception as e:
                raise RuntimeError(f"Failed to load PPO model: {e}")
            
            print(f"\nModel loaded successfully:")
            print(f"Config file: {config_path}")
            print(f"PPO model: {ppo_path}")
            if genome is not None:
                print(f"NEAT genome: {neat_path}")
            print(f"Network type: {network_type}")
            print(f"Architecture: Actor {actor_arch if 'actor_arch' in locals() else 'NEAT'}, Critic {critic_arch if 'critic_arch' in locals() else 'NEAT'}")
            
            return instance
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nUsage example:")
            print("python es_rl.py --mode play --model_path model_TIMESTAMP/config.json")
            raise
    
    def play_episode(self, max_steps=100):
        """Play one episode with rendering"""
        state, _ = self.env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).to(self.device)
            action, _ = self.ppo_network.get_action(state_tensor)
            action = action.cpu().numpy()
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
                
        return episode_reward

    def render_episodes(self, num_episodes=1):
        """Render multiple episodes to visualize agent performance"""
        print(f"\nRendering {num_episodes} episodes...")
        total_reward = 0
        
        for episode in range(num_episodes):
            reward = self.play_episode()
            total_reward += reward
            print(f"Episode {episode + 1}, Reward: {reward:.2f}")
            
        avg_reward = total_reward / num_episodes
        print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO-NEAT Mountain Car')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play'],
                      help='Mode: train or play')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to saved model config.json file (required for play mode)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs (default: 100)')
    parser.add_argument('--render_episodes', type=int, default=1,
                      help='Number of episodes to render after training (default: 1)')
    parser.add_argument('--use_es', action='store_true',
                      help='Use evolutionary strategies (NEAT) for network architecture evolution')
    args = parser.parse_args()
    
    if args.use_es:
        with open('neat_config.txt', 'w') as f:
            f.write("""[NEAT]
fitness_criterion     = max
fitness_threshold     = 100
pop_size             = 100
reset_on_extinction  = True

[DefaultGenome]
num_inputs           = 2
num_hidden          = 1
num_outputs         = 1
initial_connection  = full_direct
feed_forward        = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
activation_default  = tanh
activation_options  = tanh
activation_mutate_rate = 0.0
activation_replace_rate = 0.1
aggregation_default = sum
aggregation_options = sum
aggregation_mutate_rate = 0.0
aggregation_replace_rate = 0.1
bias_init_mean      = 0.0
bias_init_stdev     = 1.0
bias_max_value      = 30.0
bias_min_value      = -30.0
bias_replace_rate   = 0.1
bias_mutate_rate    = 0.7
bias_mutate_power   = 0.5
response_init_mean  = 1.0
response_init_stdev = 0.0
response_max_value  = 30.0
response_min_value  = -30.0
response_replace_rate = 0.1
response_mutate_rate = 0.7
response_mutate_power = 0.0
weight_init_mean    = 0.0
weight_init_stdev   = 1.0
weight_max_value    = 30.0
weight_min_value    = -30.0
weight_replace_rate = 0.1
weight_mutate_rate  = 0.8
weight_mutate_power = 0.5
enabled_default     = True
enabled_mutate_rate = 0.01
enabled_replace_rate = 0.1
node_add_prob       = 0.2
node_delete_prob   = 0.2
conn_add_prob      = 0.5
conn_delete_prob   = 0.5

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation      = 20

[DefaultReproduction]
elitism             = 2
survival_threshold  = 0.2

[DefaultNodeNames]
""")
    
    if args.mode == 'train':
        agent = PPO_NEAT('MountainCarContinuous-v0')
        rewards = agent.train(num_episodes=args.epochs, use_es=args.use_es)
        config_path = agent.save_model()
        print(f"Model saved. Config path: {config_path}")

        plot_rewards(rewards)
        agent.env = gym.make('MountainCarContinuous-v0', render_mode='human')
        agent.render_episodes(args.render_episodes)
        
    elif args.mode == 'play':
        if args.model_path is None:
            raise ValueError("Model path must be provided for play mode")
        agent = PPO_NEAT.load_model(args.model_path, render_mode='human')
        agent.render_episodes(args.render_episodes)
