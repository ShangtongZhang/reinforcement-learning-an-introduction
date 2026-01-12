"""
3D Visualization of Gridworld Value Functions
Shows value functions and policies as 3D surfaces
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter03.grid_world import step, ACTIONS, ACTION_PROB, DISCOUNT, WORLD_SIZE, A_POS, A_PRIME_POS, B_POS, B_PRIME_POS

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class Gridworld3D:
    def __init__(self):
        self.world_size = WORLD_SIZE
        
    def compute_value_function(self):
        """Compute value function using policy evaluation"""
        value = np.zeros((self.world_size, self.world_size))
        while True:
            new_value = np.zeros_like(value)
            for i in range(self.world_size):
                for j in range(self.world_size):
                    for action in ACTIONS:
                        (next_i, next_j), reward = step([i, j], action)
                        new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
            if np.sum(np.abs(value - new_value)) < 1e-4:
                break
            value = new_value
        return value
    
    def compute_optimal_value_function(self):
        """Compute optimal value function using value iteration"""
        value = np.zeros((self.world_size, self.world_size))
        while True:
            new_value = np.zeros_like(value)
            for i in range(self.world_size):
                for j in range(self.world_size):
                    values = []
                    for action in ACTIONS:
                        (next_i, next_j), reward = step([i, j], action)
                        values.append(reward + DISCOUNT * value[next_i, next_j])
                    new_value[i, j] = np.max(values)
            if np.sum(np.abs(new_value - value)) < 1e-4:
                break
            value = new_value
        return value
    
    def visualize_value_surface(self, value_func=None, title="Value Function"):
        """Visualize value function as a 3D surface"""
        if value_func is None:
            value_func = self.compute_value_function()
            
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        X, Y = np.meshgrid(range(self.world_size), range(self.world_size))
        Z = value_func
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, 
                               linewidth=0.5, antialiased=True, edgecolor='black')
        
        # Mark special states
        ax.scatter([A_POS[1]], [A_POS[0]], [value_func[A_POS[0], A_POS[1]]], 
                  c='red', s=200, marker='*', label='State A', zorder=10)
        ax.scatter([A_PRIME_POS[1]], [A_PRIME_POS[0]], [value_func[A_PRIME_POS[0], A_PRIME_POS[1]]], 
                  c='orange', s=200, marker='*', label="State A'", zorder=10)
        ax.scatter([B_POS[1]], [B_POS[0]], [value_func[B_POS[0], B_POS[1]]], 
                  c='blue', s=200, marker='*', label='State B', zorder=10)
        ax.scatter([B_PRIME_POS[1]], [B_PRIME_POS[0]], [value_func[B_PRIME_POS[0], B_PRIME_POS[1]]], 
                  c='cyan', s=200, marker='*', label="State B'", zorder=10)
        
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_zlabel('Value', fontsize=12)
        ax.set_title(f'3D {title}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.view_init(elev=45, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, f'gridworld_3d_{title.lower().replace(" ", "_")}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_policy_arrows_3d(self, value_func=None):
        """Visualize policy as 3D arrows on the value surface"""
        if value_func is None:
            value_func = self.compute_optimal_value_function()
            
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        X, Y = np.meshgrid(range(self.world_size), range(self.world_size))
        Z = value_func
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, 
                               linewidth=0.5, antialiased=True, edgecolor='black')
        
        # Compute and draw policy arrows
        action_symbols = ['←', '↑', '→', '↓']
        action_offsets = [(0, -0.3), (-0.3, 0), (0, 0.3), (0.3, 0)]
        
        for i in range(self.world_size):
            for j in range(self.world_size):
                # Find best action
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    values.append(reward + DISCOUNT * value_func[next_i, next_j])
                best_actions = [idx for idx, val in enumerate(values) if val == max(values)]
                best_action = best_actions[0]  # Take first if tie
                
                # Draw arrow
                dx, dy = action_offsets[best_action]
                ax.quiver(j, i, value_func[i, j], dx, dy, 0, 
                         color='red', arrow_length_ratio=0.3, linewidth=2)
        
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_zlabel('Value', fontsize=12)
        ax.set_title('3D Optimal Policy on Value Surface', fontsize=14, fontweight='bold')
        ax.view_init(elev=45, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'gridworld_3d_policy.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_convergence_3d(self):
        """Visualize value function convergence over iterations in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        value = np.zeros((self.world_size, self.world_size))
        iteration = 0
        value_history = []
        
        while True:
            new_value = np.zeros_like(value)
            for i in range(self.world_size):
                for j in range(self.world_size):
                    for action in ACTIONS:
                        (next_i, next_j), reward = step([i, j], action)
                        new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
            
            value_history.append(value.copy())
            if np.sum(np.abs(value - new_value)) < 1e-4:
                break
            value = new_value
            iteration += 1
            if iteration > 20:  # Limit iterations for visualization
                break
        
        # Create 3D visualization showing convergence
        n_iterations = len(value_history)
        for idx, val_func in enumerate(value_history[::max(1, n_iterations//10)]):  # Sample iterations
            X, Y = np.meshgrid(range(self.world_size), range(self.world_size))
            Z = val_func
            ax.plot_surface(X, Y, Z + idx * 2, cmap='viridis', alpha=0.6, 
                           linewidth=0.5, antialiased=True)
        
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_zlabel('Value (stacked by iteration)', fontsize=12)
        ax.set_title('3D Value Function Convergence', fontsize=14, fontweight='bold')
        ax.view_init(elev=30, azim=45)
        
        plt.savefig(os.path.join(IMAGE_DIR, 'gridworld_3d_convergence.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main function to demonstrate 3D visualizations"""
    gridworld = Gridworld3D()
    
    print("Computing and visualizing random policy value function...")
    value_func = gridworld.compute_value_function()
    gridworld.visualize_value_surface(value_func, "Random Policy Value Function")
    
    print("\nComputing and visualizing optimal value function...")
    optimal_value = gridworld.compute_optimal_value_function()
    gridworld.visualize_value_surface(optimal_value, "Optimal Value Function")
    
    print("\nVisualizing policy on value surface...")
    gridworld.visualize_policy_arrows_3d(optimal_value)
    
    print("\nVisualizing convergence...")
    gridworld.visualize_convergence_3d()


if __name__ == '__main__':
    main()
