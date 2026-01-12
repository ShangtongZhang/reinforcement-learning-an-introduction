"""
3D Interactive Visualization of Cliff Walking
Shows the agent's path and Q-values in 3D
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter06.cliff_walking import step, ACTIONS, START, GOAL, WORLD_HEIGHT, WORLD_WIDTH

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class CliffWalking3D:
    def __init__(self):
        self.world_height = WORLD_HEIGHT
        self.world_width = WORLD_WIDTH
        self.start = START
        self.goal = GOAL
        
    def visualize_environment_3d(self):
        """Visualize the cliff walking environment in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid
        X, Y = np.meshgrid(range(self.world_width), range(self.world_height))
        Z = np.zeros_like(X)
        
        # Mark cliff area (row 2, columns 1-10)
        cliff_mask = np.zeros_like(Z, dtype=bool)
        cliff_mask[2, 1:11] = True
        
        # Create surface with different colors for cliff
        colors = np.ones_like(Z)
        colors[cliff_mask] = 0.3  # Darker for cliff
        
        # Plot surface
        ax.plot_surface(X, Y, Z, facecolors=plt.cm.RdYlGn(colors), 
                       alpha=0.7, linewidth=0.5, antialiased=True)
        
        # Mark start position
        ax.scatter([self.start[1]], [self.start[0]], [0.1], 
                  c='green', s=300, marker='o', label='Start', zorder=10)
        
        # Mark goal position
        ax.scatter([self.goal[1]], [self.goal[0]], [0.1], 
                  c='blue', s=300, marker='*', label='Goal', zorder=10)
        
        # Mark cliff with red bars
        for j in range(1, 11):
            ax.bar3d(j, 2, 0, 0.8, 0.8, -0.5, color='red', alpha=0.8, label='Cliff' if j == 1 else '')
        
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_zlabel('Height', fontsize=12)
        ax.set_title('3D Cliff Walking Environment', fontsize=14, fontweight='bold')
        ax.set_xlim([-0.5, self.world_width - 0.5])
        ax.set_ylim([-0.5, self.world_height - 0.5])
        ax.set_zlim([-0.6, 0.5])
        ax.legend()
        ax.view_init(elev=60, azim=45)
        
        plt.savefig(os.path.join(IMAGE_DIR, 'cliff_walking_3d_environment.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def simulate_episode(self, q_value, epsilon=0.1):
        """Simulate one episode and return the path"""
        state = self.start.copy()
        path = [state.copy()]
        total_reward = 0
        
        while state != self.goal:
            # Choose action
            if np.random.rand() < epsilon:
                action = np.random.choice(ACTIONS)
            else:
                values = q_value[state[0], state[1], :]
                action = np.random.choice([a for a, v in enumerate(values) if v == np.max(values)])
            
            # Take step
            next_state, reward = step(state, action)
            path.append(next_state.copy())
            total_reward += reward
            state = next_state
            
            if len(path) > 1000:  # Safety limit
                break
                
        return path, total_reward
        
    def visualize_path_3d(self, path, title="Agent Path"):
        """Visualize agent's path in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create environment
        X, Y = np.meshgrid(range(self.world_width), range(self.world_height))
        Z = np.zeros_like(X)
        cliff_mask = np.zeros_like(Z, dtype=bool)
        cliff_mask[2, 1:11] = True
        colors = np.ones_like(Z)
        colors[cliff_mask] = 0.3
        
        ax.plot_surface(X, Y, Z, facecolors=plt.cm.RdYlGn(colors), 
                       alpha=0.5, linewidth=0.5, antialiased=True)
        
        # Extract path coordinates
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            path_z = [0.1 + i * 0.01 for i in range(len(path))]  # Slight elevation
            
            # Plot path
            ax.plot(path_x, path_y, path_z, 'b-', linewidth=3, label='Agent Path', zorder=5)
            ax.scatter(path_x, path_y, path_z, c=range(len(path)), 
                      cmap='cool', s=50, alpha=0.8, zorder=6)
            
            # Mark start and end
            ax.scatter([path_x[0]], [path_y[0]], [path_z[0]], 
                      c='green', s=300, marker='o', label='Start', zorder=10)
            ax.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], 
                      c='red', s=300, marker='*', label='End', zorder=10)
        
        # Mark cliff
        for j in range(1, 11):
            ax.bar3d(j, 2, 0, 0.8, 0.8, -0.5, color='red', alpha=0.6)
        
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_zlabel('Time Step', fontsize=12)
        ax.set_title(f'3D {title}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.view_init(elev=60, azim=45)
        
        plt.savefig(os.path.join(IMAGE_DIR, f'cliff_walking_3d_{title.lower().replace(" ", "_")}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_q_values_3d(self, q_value):
        """Visualize Q-values as 3D surface for each action"""
        fig = plt.figure(figsize=(18, 12))
        
        action_names = ['Up', 'Down', 'Left', 'Right']
        
        for action_idx, action_name in enumerate(action_names):
            ax = fig.add_subplot(2, 2, action_idx + 1, projection='3d')
            
            # Extract Q-values for this action
            Q_action = q_value[:, :, action_idx]
            
            # Create meshgrid
            X, Y = np.meshgrid(range(self.world_width), range(self.world_height))
            Z = Q_action
            
            # Plot surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                                 linewidth=0.5, antialiased=True, edgecolor='black')
            
            # Mark special positions
            ax.scatter([self.start[1]], [self.start[0]], 
                      [Q_action[self.start[0], self.start[1]]], 
                      c='green', s=200, marker='o', zorder=10)
            ax.scatter([self.goal[1]], [self.goal[0]], 
                      [Q_action[self.goal[0], self.goal[1]]], 
                      c='red', s=200, marker='*', zorder=10)
            
            ax.set_xlabel('Column', fontsize=10)
            ax.set_ylabel('Row', fontsize=10)
            ax.set_zlabel('Q-Value', fontsize=10)
            ax.set_title(f'Q-Values for Action: {action_name}', fontsize=12, fontweight='bold')
            ax.view_init(elev=45, azim=45)
            
            plt.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'cliff_walking_3d_q_values.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main function to demonstrate 3D visualizations"""
    cliff = CliffWalking3D()
    
    print("Visualizing environment...")
    cliff.visualize_environment_3d()
    
    # Create a sample Q-value table (would normally come from training)
    print("\nCreating sample Q-values...")
    q_value = np.random.randn(WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)) * 0.1
    
    # Set goal to have high values
    q_value[GOAL[0], GOAL[1], :] = 10
    
    print("\nVisualizing Q-values...")
    cliff.visualize_q_values_3d(q_value)
    
    # Simulate a path
    print("\nSimulating episode...")
    path, reward = cliff.simulate_episode(q_value, epsilon=0.1)
    print(f"Episode completed with {len(path)} steps, total reward: {reward}")
    
    print("\nVisualizing path...")
    cliff.visualize_path_3d(path, "Sample Agent Path")


if __name__ == '__main__':
    main()
