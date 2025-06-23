"""
Demo script for PPO-NES hybrid algorithm on Atari Assault
This script shows how to use the PPO-NES implementation with different configurations
"""

import numpy as np
from ppo_nes import PPONESAgent
from datetime import datetime


def demo_quick_training():
    """Quick demo with reduced parameters for faster testing"""
    print("=" * 60)
    print("QUICK DEMO: PPO-NES on Atari Assault")
    print("=" * 60)

    agent = PPONESAgent(
        env_name="ALE/Assault-v5",
        lr=5e-4,
        gamma=0.99,
        clip_ratio=0.2,
        update_epochs=3,
        batch_size=32,
        buffer_size=64,
        nes_population_size=10,
        nes_sigma=0.05,
        nes_frequency=5
    )

    # Quick training for demo
    agent.train(total_timesteps=5000, log_interval=2)

    # Plot and save results
    agent.plot_results("demo_quick_results.png")
    agent.save_model("demo_quick_model.pth")

    return agent


def demo_standard_training():
    """Standard training configuration"""
    print("=" * 60)
    print("STANDARD TRAINING: PPO-NES on Atari Assault")
    print("=" * 60)

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

    # Standard training
    agent.train(total_timesteps=25000, log_interval=5)

    # Plot and save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.plot_results(f"standard_training_results_{timestamp}.png")
    agent.save_model(f"standard_model_{timestamp}.pth")

    return agent


def demo_extended_training():
    """Extended training for better performance"""
    print("=" * 60)
    print("EXTENDED TRAINING: PPO-NES on Atari Assault")
    print("=" * 60)

    agent = PPONESAgent(
        env_name="ALE/Assault-v5",
        lr=2e-4,
        gamma=0.99,
        clip_ratio=0.1,
        update_epochs=4,
        batch_size=64,
        buffer_size=256,
        nes_population_size=20,
        nes_sigma=0.015,
        nes_frequency=10
    )

    # Extended training
    agent.train(total_timesteps=100000, log_interval=10)

    # Plot and save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.plot_results(f"extended_training_results_{timestamp}.png")
    agent.save_model(f"extended_model_{timestamp}.pth")

    return agent


def demo_nes_focused():
    """Demo with more frequent NES evolution"""
    print("=" * 60)
    print("NES-FOCUSED TRAINING: More Evolutionary Steps")
    print("=" * 60)

    agent = PPONESAgent(
        env_name="ALE/Assault-v5",
        lr=3e-4,
        gamma=0.99,
        clip_ratio=0.15,
        update_epochs=3,
        batch_size=32,
        buffer_size=128,
        nes_population_size=25,
        nes_sigma=0.03,
        nes_frequency=3  # More frequent NES evolution
    )

    # Training with frequent evolution
    agent.train(total_timesteps=20000, log_interval=3)

    # Plot and save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.plot_results(f"nes_focused_results_{timestamp}.png")
    agent.save_model(f"nes_focused_model_{timestamp}.pth")

    return agent


def analyze_results(agent):
    """Analyze and print training statistics"""
    print("\n" + "=" * 60)
    print("TRAINING ANALYSIS")
    print("=" * 60)

    if agent.episode_rewards:
        rewards = list(agent.episode_rewards)
        print(f"Episode Statistics:")
        print(f"  Total Episodes: {len(rewards)}")
        print(f"  Average Reward: {np.mean(rewards):.2f}")
        print(f"  Best Reward: {np.max(rewards):.2f}")
        print(f"  Worst Reward: {np.min(rewards):.2f}")
        print(f"  Reward Std: {np.std(rewards):.2f}")

    if agent.nes_rewards:
        nes_rewards = agent.nes_rewards
        print(f"\nNES Evolution Statistics:")
        print(f"  Total Generations: {len(nes_rewards)}")
        print(f"  Best NES Fitness: {np.max(nes_rewards):.2f}")
        print(f"  Latest NES Fitness: {nes_rewards[-1]:.2f}")
        print(f"  NES Improvement: {nes_rewards[-1] - nes_rewards[0]:.2f}")

    if agent.training_rewards:
        ppo_rewards = agent.training_rewards
        print(f"\nPPO Training Progress:")
        print(f"  Training Updates: {len(ppo_rewards)}")
        print(f"  Initial Avg Reward: {ppo_rewards[0]:.2f}")
        print(f"  Final Avg Reward: {ppo_rewards[-1]:.2f}")
        print(f"  PPO Improvement: {ppo_rewards[-1] - ppo_rewards[0]:.2f}")


def compare_algorithms():
    """Compare different algorithm configurations"""
    print("=" * 60)
    print("ALGORITHM COMPARISON")
    print("=" * 60)

    # Test different NES frequencies
    configs = [
        {"name": "PPO-Heavy", "nes_frequency": 20, "nes_pop": 10},
        {"name": "Balanced", "nes_frequency": 8, "nes_pop": 15},
        {"name": "NES-Heavy", "nes_frequency": 3, "nes_pop": 20},
    ]

    results = []

    for config in configs:
        print(f"\nTesting: {config['name']}")

        agent = PPONESAgent(
            env_name="ALE/Assault-v5",
            lr=3e-4,
            gamma=0.99,
            clip_ratio=0.15,
            update_epochs=3,
            batch_size=32,
            buffer_size=64,
            nes_population_size=config["nes_pop"],
            nes_sigma=0.02,
            nes_frequency=config["nes_frequency"]
        )

        agent.train(total_timesteps=8000, log_interval=5)

        # Calculate performance metrics
        final_reward = np.mean(list(agent.episode_rewards)
                               ) if agent.episode_rewards else 0
        best_nes = np.max(agent.nes_rewards) if agent.nes_rewards else 0

        results.append({
            "name": config["name"],
            "final_reward": final_reward,
            "best_nes": best_nes
        })

        print(f"Final Avg Reward: {final_reward:.2f}")
        print(f"Best NES Fitness: {best_nes:.2f}")

    # Print comparison
    print("\n" + "=" * 40)
    print("COMPARISON RESULTS")
    print("=" * 40)
    for result in results:
        print(
            f"{result['name']:12} | Reward: {result['final_reward']:6.2f} | NES: {result['best_nes']:6.2f}")


def main():
    """Main demo function"""
    print("PPO-NES Hybrid Algorithm Demo")
    print("Choose a demo mode:")
    print("1. Quick Demo (5k timesteps)")
    print("2. Standard Training (25k timesteps)")
    print("3. Extended Training (100k timesteps)")
    print("4. NES-Focused Training (20k timesteps)")
    print("5. Algorithm Comparison")

    try:
        choice = input(
            "Enter choice (1-5) or press Enter for quick demo: ").strip()
        if not choice:
            choice = "1"
    except KeyboardInterrupt:
        print("\nExiting...")
        return

    if choice == "1":
        agent = demo_quick_training()
        analyze_results(agent)
    elif choice == "2":
        agent = demo_standard_training()
        analyze_results(agent)
    elif choice == "3":
        agent = demo_extended_training()
        analyze_results(agent)
    elif choice == "4":
        agent = demo_nes_focused()
        analyze_results(agent)
    elif choice == "5":
        compare_algorithms()
    else:
        print("Invalid choice, running quick demo...")
        agent = demo_quick_training()
        analyze_results(agent)

    print("\nDemo completed! Check the generated plots and model files.")


if __name__ == "__main__":
    main()
