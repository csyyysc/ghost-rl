import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class BlackjackMC:
    def __init__(self):
        self.deck = self._create_deck()
        self.action_values = defaultdict(lambda: {'hit': 0, 'stand': 0})
        self.action_counts = defaultdict(lambda: {'hit': 0, 'stand': 0})
        self.episode_rewards = []
        
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
    
    def play_episode(self):
        """Play one episode of blackjack"""
        player_hand = [self._draw_card(), self._draw_card()]
        dealer_hand = [self._draw_card(), self._draw_card()]
        dealer_card = dealer_hand[0]  # Only one dealer card is visible
        
        states = []
        actions = []
        rewards = []
        
        while True:
            state = self._get_state(player_hand, dealer_card)
            states.append(state)
            
            # Choose action (hit or stand) based on current estimates
            if self.action_counts[state]['hit'] == 0 and \
               self.action_counts[state]['stand'] == 0:
                action = np.random.choice(['hit', 'stand'])
            else:
                hit_value = self.action_values[state]['hit'] \
                    / max(1, self.action_counts[state]['hit'])
                stand_value = self.action_values[state]['stand'] \
                    / max(1, self.action_counts[state]['stand'])
                action = 'hit' if hit_value > stand_value else 'stand'
            
            actions.append(action)
            
            if action == 'hit':
                player_hand.append(self._draw_card())
                if self._is_bust(player_hand):
                    rewards.append(-1) # Player loses
                    break
            else:
                dealer_hand = self._dealer_play(dealer_hand)
                dealer_value = self._get_hand_value(dealer_hand)
                player_value = self._get_hand_value(player_hand)
                
                if self._is_bust(dealer_hand):
                    rewards.append(1)  # Dealer busts, player wins
                elif dealer_value > player_value:
                    rewards.append(-1)  # Dealer wins
                elif dealer_value < player_value:
                    rewards.append(1)  # Player wins
                else:
                    rewards.append(0)  # Draw
                break
        
        # Update action values using Monte Carlo method
        for i, (state, action) in enumerate(zip(states, actions)):
            G = sum(rewards[i:])
            self.action_values[state][action] += G
            self.action_counts[state][action] += 1
        
        # Store the final reward for this episode
        self.episode_rewards.append(rewards[-1])
    
    def get_policy(self):
        """Get the current policy (best action for each state)"""
        policy = {}
        for state in self.action_values:
            hit_value = self.action_values[state]['hit'] \
                / max(1, self.action_counts[state]['hit'])
            stand_value = self.action_values[state]['stand'] \
                / max(1, self.action_counts[state]['stand'])
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
                hit_values[player_val, dealer_val] = (
                    self.action_values[state]['hit'] / 
                    max(1, self.action_counts[state]['hit'])
                )
                stand_values[player_val, dealer_val] = (
                    self.action_values[state]['stand'] / 
                    max(1, self.action_counts[state]['stand'])
                )
        
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
    agent = BlackjackMC()
    
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
