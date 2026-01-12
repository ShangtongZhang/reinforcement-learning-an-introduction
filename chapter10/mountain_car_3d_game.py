"""
3D Interactive Mountain Car Game
Play the mountain car game in 3D with real-time visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter10.mountain_car import step, ACTIONS, POSITION_MIN, POSITION_MAX, VELOCITY_MIN, VELOCITY_MAX

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class MountainCar3DGame:
    def __init__(self):
        self.position = -0.5
        self.velocity = 0.0
        self.steps = 0
        self.max_steps = 1000
        self.history = []
        
    def reset(self):
        """Reset the game"""
        self.position = np.random.uniform(-0.6, -0.4)
        self.velocity = 0.0
        self.steps = 0
        self.history = []
        
    def get_state(self):
        """Get current state"""
        return self.position, self.velocity
        
    def take_action(self, action):
        """Take an action and return reward"""
        self.position, self.velocity, reward = step(self.position, self.velocity, action)
        self.steps += 1
        self.history.append((self.position, self.velocity, reward))
        return reward, self.is_done()
        
    def is_done(self):
        """Check if episode is done"""
        return self.position >= POSITION_MAX or self.steps >= self.max_steps
        
    def visualize_environment_3d(self):
        """Visualize the mountain car environment in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mountain landscape
        x = np.linspace(POSITION_MIN, POSITION_MAX, 100)
        y = np.sin(3 * x)  # Mountain shape
        
        # Create 3D surface
        X = np.linspace(POSITION_MIN, POSITION_MAX, 50)
        Y = np.linspace(VELOCITY_MIN, VELOCITY_MAX, 50)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(3 * X)  # Mountain surface
        
        # Plot mountain surface
        surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.7, 
                               linewidth=0.5, antialiased=True)
        
        # Mark goal position
        goal_x = POSITION_MAX
        goal_y = 0
        goal_z = np.sin(3 * goal_x)
        ax.scatter([goal_x], [goal_y], [goal_z], 
                  c='green', s=500, marker='*', label='Goal', zorder=10)
        
        # Mark start position
        start_x = -0.5
        start_y = 0
        start_z = np.sin(3 * start_x)
        ax.scatter([start_x], [start_y], [start_z], 
                  c='red', s=500, marker='o', label='Start', zorder=10)
        
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Velocity', fontsize=12)
        ax.set_zlabel('Height', fontsize=12)
        ax.set_title('3D Mountain Car Environment', fontsize=14, fontweight='bold')
        ax.legend()
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'mountain_car_3d_environment.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_trajectory_3d(self, trajectory=None):
        """Visualize a trajectory in 3D"""
        if trajectory is None:
            trajectory = self.history
            
        if not trajectory:
            print("No trajectory to visualize")
            return
            
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mountain surface
        X = np.linspace(POSITION_MIN, POSITION_MAX, 50)
        Y = np.linspace(VELOCITY_MIN, VELOCITY_MAX, 50)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(3 * X)
        
        ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.5, 
                       linewidth=0.5, antialiased=True)
        
        # Extract trajectory
        positions = [t[0] for t in trajectory]
        velocities = [t[1] for t in trajectory]
        heights = [np.sin(3 * p) for p in positions]
        rewards = [t[2] for t in trajectory]
        
        # Plot trajectory
        ax.plot(positions, velocities, heights, 'b-', linewidth=3, label='Trajectory', zorder=5)
        scatter = ax.scatter(positions, velocities, heights, 
                           c=range(len(trajectory)), cmap='cool', 
                           s=100, alpha=0.8, zorder=6)
        
        # Mark start and end
        ax.scatter([positions[0]], [velocities[0]], [heights[0]], 
                  c='green', s=300, marker='o', label='Start', zorder=10)
        ax.scatter([positions[-1]], [velocities[-1]], [heights[-1]], 
                  c='red', s=300, marker='*', label='End', zorder=10)
        
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Velocity', fontsize=12)
        ax.set_zlabel('Height', fontsize=12)
        ax.set_title('3D Mountain Car Trajectory', fontsize=14, fontweight='bold')
        ax.legend()
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'mountain_car_3d_trajectory.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_value_function_3d(self, value_func):
        """Visualize value function as 3D surface"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid
        positions = np.linspace(POSITION_MIN, POSITION_MAX, 50)
        velocities = np.linspace(VELOCITY_MIN, VELOCITY_MAX, 50)
        P, V = np.meshgrid(positions, velocities)
        
        # Compute values (simplified - would need actual value function)
        # For demonstration, create a sample value function
        Z = np.zeros_like(P)
        for i in range(len(positions)):
            for j in range(len(velocities)):
                # Sample value function (would be replaced with actual)
                Z[j, i] = -np.abs(positions[i] - POSITION_MAX) * 10
        
        # Plot surface
        surf = ax.plot_surface(P, V, Z, cmap='viridis', alpha=0.8, 
                               linewidth=0.5, antialiased=True)
        
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Velocity', fontsize=12)
        ax.set_zlabel('Value', fontsize=12)
        ax.set_title('3D Value Function Surface', fontsize=14, fontweight='bold')
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'mountain_car_3d_value.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def play_random_episode(self):
        """Play a random episode"""
        self.reset()
        while not self.is_done():
            action = np.random.choice(ACTIONS)
            reward, done = self.take_action(action)
            if done:
                break
        return self.history


def main():
    """Main function to demonstrate 3D visualizations"""
    game = MountainCar3DGame()
    
    print("Visualizing environment...")
    game.visualize_environment_3d()
    
    print("\nPlaying random episode...")
    trajectory = game.play_random_episode()
    print(f"Episode completed in {len(trajectory)} steps")
    
    print("\nVisualizing trajectory...")
    game.visualize_trajectory_3d(trajectory)
    
    print("\nVisualizing value function...")
    # Create sample value function
    value_func = np.zeros((50, 50))
    game.visualize_value_function_3d(value_func)


if __name__ == '__main__':
    main()
