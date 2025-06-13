import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class BlackjackTD:
    def __init__(self, alpha=0.1, gamma=1.0):
        self.deck = self._create_deck()
        self.action_values = defaultdict(lambda: {'hit': 0, 'stand': 0})
        self.action_counts = defaultdict(lambda: {'hit': 0, 'stand': 0})
        self.episode_rewards = []
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        
    def _create_deck(self):
        """Create a deck of cards (1-10, with face cards as 10)"""
        return list(range(1, 11)) * 4  # 4 suits
    
    def _draw_card(self):
        """Draw a random card from the deck"""
        return np.random.choice(self.deck)
    
    def _get_hand_value(self, hand):
        """Calculate the value of a hand, handling aces (1 or 11)"""
        value = sum(hand)
        if 1 in hand and value + 10 <= 21:
            value += 10
        return value
    
    def _is_bust(self, hand):
        """Check if a hand is bust (over 21)"""
        return self._get_hand_value(hand) > 21
    
    def _dealer_play(self, dealer_hand):
        """Dealer plays according to standard rules (hit on 16 or less)"""
        while self._get_hand_value(dealer_hand) < 17:
            dealer_hand.append(self._draw_card())
        return dealer_hand
    
    def _get_state(self, player_hand, dealer_card):
        """Get the current state of the game"""
        return (self._get_hand_value(player_hand), dealer_card)
    
    def _get_action_value(self, state, action):
        """Get the current value estimate for a state-action pair"""
        return self.action_values[state][action] \
            / max(1, self.action_counts[state][action])
    
    def _choose_action(self, state, epsilon=0.1):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < epsilon:
            return np.random.choice(['hit', 'stand'])
        
        hit_value = self._get_action_value(state, 'hit')
        stand_value = self._get_action_value(state, 'stand')
        return 'hit' if hit_value > stand_value else 'stand'
    
    def play_episode(self):
        """Play one episode of blackjack using TD(0) learning"""
        player_hand = [self._draw_card(), self._draw_card()]
        dealer_hand = [self._draw_card(), self._draw_card()]
        dealer_card = dealer_hand[0]
        
        current_state = self._get_state(player_hand, dealer_card)
        episode_reward = 0
        
        while True:
            # Choose action using epsilon-greedy policy
            action = self._choose_action(current_state)
            self.action_counts[current_state][action] += 1
            
            # Take action and observe next state and reward
            if action == 'hit':
                player_hand.append(self._draw_card())
                if self._is_bust(player_hand):
                    reward = -1
                    next_state = None  # Terminal state
                else:
                    reward = 0
                    next_state = self._get_state(player_hand, dealer_card)
            else:
                dealer_hand = self._dealer_play(dealer_hand)
                dealer_value = self._get_hand_value(dealer_hand)
                player_value = self._get_hand_value(player_hand)
                
                if self._is_bust(dealer_hand):
                    reward = 1
                elif dealer_value > player_value:
                    reward = -1
                elif dealer_value < player_value:
                    reward = 1
                else:
                    reward = 0
                next_state = None 
            
            current_value = self._get_action_value(current_state, action)
            if next_state is None:
                next_value = 0
            else:
                # Choose best action for next state
                next_action = self._choose_action(next_state, epsilon=0)
                next_value = \
                    self._get_action_value(next_state, next_action)
            
            # Update value estimate
            td_target = reward + self.gamma * next_value
            td_error = td_target - current_value
            self.action_values[current_state][action] += self.alpha * td_error
            
            episode_reward += reward
            
            if next_state is None:
                break
                
            current_state = next_state
        
        self.episode_rewards.append(episode_reward)
    
    def get_policy(self):
        """Get the current policy (best action for each state)"""
        policy = {}
        for state in self.action_values:
            hit_value = self._get_action_value(state, 'hit')
            stand_value = self._get_action_value(state, 'stand')
            policy[state] = 'hit' if hit_value > stand_value else 'stand'
        return policy
    
    def plot_learning_curve(self):
        """Plot the learning curve (average reward over time)"""
        plt.figure(figsize=(10, 5))
        window_size = 100
        moving_avg = np.convolve(self.episode_rewards, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        
        plt.plot(moving_avg)
        plt.title('Learning Curve (Moving Average of Rewards)')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.show()
    
    def plot_value_function(self):
        """Plot the value function as a heatmap"""
        max_player = 21
        max_dealer = 10
        hit_values = np.zeros((max_player + 1, max_dealer + 1))
        stand_values = np.zeros((max_player + 1, max_dealer + 1))
        
        for state in self.action_values:
            player_val, dealer_val = state
            if player_val <= max_player and dealer_val <= max_dealer:
                hit_values[player_val, dealer_val] = self._get_action_value(state, 'hit')
                stand_values[player_val, dealer_val] = self._get_action_value(state, 'stand')
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(hit_values, cmap='RdYlGn', center=0)
        plt.title('Value Function for Hit')
        plt.xlabel('Dealer Card')
        plt.ylabel('Player Sum')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(stand_values, cmap='RdYlGn', center=0)
        plt.title('Value Function for Stand')
        plt.xlabel('Dealer Card')
        plt.ylabel('Player Sum')
        
        plt.tight_layout()
        plt.show()
    
    def plot_policy(self):
        """Plot the learned policy"""
        policy = self.get_policy()
        max_player = 21
        max_dealer = 10
        policy_matrix = np.zeros((max_player + 1, max_dealer + 1))

        for state, action in policy.items():
            player_val, dealer_val = state
            if player_val <= max_player and dealer_val <= max_dealer:
                policy_matrix[player_val, dealer_val] = 1 if action == 'hit' else 0
        
        plt.figure(figsize=(8, 8))
        sns.heatmap(policy_matrix, cmap='RdYlGn', center=0.5)
        plt.title('Learned Policy (1=Hit, 0=Stand)')
        plt.xlabel('Dealer Card')
        plt.ylabel('Player Sum')
        plt.show()

def main():
    n_episodes = 10000
    agent = BlackjackTD(alpha=0.1, gamma=0.99)
    
    print("Training the agent...")
    for i in range(n_episodes):
        agent.play_episode()
        if (i + 1) % 1000 == 0:
            print(f"Completed {i + 1} episodes")
    
    policy = agent.get_policy()
    print("\nLearned Policy "
          "(Player's Hand Value, Dealer's Card) -> Action:")
    for state, action in sorted(policy.items()):
        print(f"State {state[0]}, {state[1]}: {action}")

    print("\nGenerating plots...")
    agent.plot_learning_curve()
    agent.plot_value_function()
    agent.plot_policy()

if __name__ == "__main__":
    main()
