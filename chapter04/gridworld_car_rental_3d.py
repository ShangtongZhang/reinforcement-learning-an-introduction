"""
3D Visualizations for Chapter 4: Dynamic Programming
Includes Grid World, Car Rental, and Gambler's Problem
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))


class CarRental3D:
    """3D Visualization of Jack's Car Rental Problem"""
    
    def __init__(self, max_cars=20):
        self.max_cars = max_cars
        
    def visualize_policy_3d(self, policy):
        """Visualize car rental policy as 3D surface"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        X, Y = np.meshgrid(range(self.max_cars + 1), range(self.max_cars + 1))
        Z = policy
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.9,
                               linewidth=0.5, antialiased=True, edgecolor='black')
        
        ax.set_xlabel('# Cars at Location 1', fontsize=12)
        ax.set_ylabel('# Cars at Location 2', fontsize=12)
        ax.set_zlabel('# Cars to Move', fontsize=12)
        ax.set_title('3D Car Rental Policy', fontsize=14, fontweight='bold')
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'car_rental_3d_policy.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_value_function_3d(self, values):
        """Visualize value function as 3D surface"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        X, Y = np.meshgrid(range(self.max_cars + 1), range(self.max_cars + 1))
        Z = values
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9,
                               linewidth=0.5, antialiased=True, edgecolor='black')
        
        ax.set_xlabel('# Cars at Location 1', fontsize=12)
        ax.set_ylabel('# Cars at Location 2', fontsize=12)
        ax.set_zlabel('Expected Return', fontsize=12)
        ax.set_title('3D Car Rental Value Function', fontsize=14, fontweight='bold')
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'car_rental_3d_value.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


class GamblerProblem3D:
    """3D Visualization of Gambler's Problem"""
    
    def __init__(self, goal=100):
        self.goal = goal
        
    def visualize_value_function_3d(self, state_values):
        """Visualize value function and policy as 3D surface"""
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Value function
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        states = np.arange(1, self.goal)
        values = state_values[1:-1]
        
        # Create mesh for surface
        X = states
        Y = np.zeros_like(X)
        
        # Plot as 3D bars
        for i, (state, value) in enumerate(zip(states, values)):
            ax1.bar3d(state, 0, 0, 0.8, 0.8, value, 
                     color=plt.cm.viridis(value), alpha=0.8)
        
        ax1.set_xlabel('Capital', fontsize=12)
        ax1.set_ylabel('Iteration', fontsize=12)
        ax1.set_zlabel('Value', fontsize=12)
        ax1.set_title('3D Value Function', fontsize=14, fontweight='bold')
        ax1.view_init(elev=30, azim=45)
        
        # Plot 2: Optimal policy
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        # Compute optimal policy
        policy = np.zeros(self.goal + 1)
        for state in range(1, self.goal):
            actions = np.arange(min(state, self.goal - state) + 1)
            policy[state] = actions[0]  # Simplified
        
        for i, (state, stake) in enumerate(zip(states, policy[1:-1])):
            ax2.bar3d(state, 0, 0, 0.8, 0.8, stake,
                     color=plt.cm.plasma(stake / max(policy)), alpha=0.8)
        
        ax2.set_xlabel('Capital', fontsize=12)
        ax2.set_ylabel('Iteration', fontsize=12)
        ax2.set_zlabel('Optimal Stake', fontsize=12)
        ax2.set_title('3D Optimal Policy', fontsize=14, fontweight='bold')
        ax2.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'gambler_3d_combined.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main function to demonstrate 3D visualizations"""
    
    # Car Rental
    print("Creating Car Rental visualizations...")
    car_rental = CarRental3D(max_cars=20)
    
    # Create sample policy (would normally come from training)
    sample_policy = np.zeros((21, 21))
    for i in range(21):
        for j in range(21):
            sample_policy[i, j] = (i - j) / 2  # Move cars from high to low
    
    # Create sample value function
    sample_values = np.random.randn(21, 21) * 50 + 100
    
    car_rental.visualize_policy_3d(sample_policy)
    car_rental.visualize_value_function_3d(sample_values)
    
    # Gambler's Problem
    print("\nCreating Gambler's Problem visualizations...")
    gambler = GamblerProblem3D(goal=100)
    
    # Create sample value function
    state_values = np.zeros(101)
    for i in range(1, 100):
        state_values[i] = i / 100.0  # Simplified
    
    gambler.visualize_value_function_3d(state_values)
    
    print("\nAll Chapter 4 visualizations created!")


if __name__ == '__main__':
    main()
