import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class MonteCarloOptionPricing:
    def __init__(self, 
                 spot_price: float,
                 strike_price: float,
                 risk_free_rate: float = 0.015,  # Taiwan's typical risk-free rate
                 volatility: float = 0.18,       # Typical TXO volatility
                 time_to_maturity: float = 1.0,
                 num_simulations: int = 10000,
                 num_steps: int = 240):          # Taiwan trading days
        """
        Initialize the Monte Carlo option pricing model for Taiwan index options (TXO).
        
        Args:
            spot_price: Current TAIEX index level
            strike_price: Option strike price in index points
            risk_free_rate: Annual risk-free rate (default: 1.5%)
            volatility: Annual volatility (default: 18% - typical for TXO)
            time_to_maturity: Time to expiration in years
            num_simulations: Number of Monte Carlo simulations
            num_steps: Number of trading days (default: 240 for Taiwan market)
        """
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.time_to_maturity = time_to_maturity
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        
    def generate_price_paths(self) -> np.ndarray:
        """
        Generate index price paths using Geometric Brownian Motion.
        Adjusted for Taiwan market characteristics.
        """
        dt = self.time_to_maturity / self.num_steps
        drift = (self.risk_free_rate - 0.5 * self.volatility**2) * dt
        diffusion = self.volatility * np.sqrt(dt)
        
        # Generate random walks
        random_walks = np.random.normal(0, 1, (self.num_simulations, self.num_steps))
        
        # Calculate price paths
        price_paths = np.zeros((self.num_simulations, self.num_steps + 1))
        price_paths[:, 0] = self.spot_price
        
        for t in range(1, self.num_steps + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion * random_walks[:, t-1])
            
        return price_paths
    
    def price_european_call(self) -> Tuple[float, float]:
        """
        Price a European call option using Monte Carlo simulation.
        Returns price in index points.
        """
        price_paths = self.generate_price_paths()
        final_prices = price_paths[:, -1]
        
        # Calculate payoffs
        payoffs = np.maximum(final_prices - self.strike_price, 0)
        
        # Discount payoffs
        discounted_payoffs = payoffs * np.exp(-self.risk_free_rate * self.time_to_maturity)
        
        # Calculate option price and standard error
        option_price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs) / np.sqrt(self.num_simulations)
        
        return option_price, standard_error
    
    def plot_price_paths(self, num_paths: int = 10) -> None:
        """
        Plot a sample of price paths with Taiwan market context.
        """
        price_paths = self.generate_price_paths()
        time_points = np.linspace(0, self.time_to_maturity, self.num_steps + 1)
        
        plt.figure(figsize=(12, 6))
        for i in range(min(num_paths, self.num_simulations)):
            plt.plot(time_points, price_paths[i], alpha=0.5)
        
        plt.title('TAIEX Index Paths Simulation')
        plt.xlabel('Time (years)')
        plt.ylabel('Index Level')
        plt.grid(True)
        
        # Add current price line
        plt.axhline(y=self.spot_price, color='r', linestyle='--', label='Current Index Level')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Example parameters for Taiwan index options (TXO)
    spot_price = 18000.0    # Example TAIEX level
    strike_price = 18000.0  # At-the-money option
    risk_free_rate = 0.015  # 1.5% - typical Taiwan rate
    volatility = 0.18       # 18% - typical for TXO
    time_to_maturity = 1.0  # 1 year
    
    mc_model = MonteCarloOptionPricing(
        spot_price=spot_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_maturity=time_to_maturity
    )
    
    option_price, standard_error = mc_model.price_european_call()
    print(f"TXO Call Option Price: {option_price:.1f} points")
    print(f"Standard Error: {standard_error:.1f} points")
    
    mc_model.plot_price_paths()
