"""
3D Visualization of Maze Exploration
Shows maze structure and agent exploration in 3D
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter08.maze import Maze

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class Maze3D:
    def __init__(self, maze=None):
        if maze is None:
            self.maze = Maze()
        else:
            self.maze = maze
            
    def visualize_maze_3d(self):
        """Visualize the maze structure in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid
        X, Y = np.meshgrid(range(self.maze.WORLD_WIDTH), range(self.maze.WORLD_HEIGHT))
        Z = np.zeros_like(X)
        
        # Mark obstacles
        obstacle_mask = np.zeros_like(Z, dtype=bool)
        for obs in self.maze.obstacles:
            obstacle_mask[obs[0], obs[1]] = True
        
        # Create surface with different colors
        colors = np.ones_like(Z)
        colors[obstacle_mask] = 0.3  # Darker for obstacles
        
        # Plot surface
        ax.plot_surface(X, Y, Z, facecolors=plt.cm.RdYlGn(colors), 
                       alpha=0.7, linewidth=0.5, antialiased=True)
        
        # Draw obstacles as 3D bars
        for obs in self.maze.obstacles:
            ax.bar3d(obs[1], obs[0], 0, 0.8, 0.8, 1.0, 
                    color='red', alpha=0.8, label='Obstacle' if obs == self.maze.obstacles[0] else '')
        
        # Mark start position
        ax.scatter([self.maze.START_STATE[1]], [self.maze.START_STATE[0]], [0.1], 
                  c='green', s=500, marker='o', label='Start', zorder=10)
        
        # Mark goal positions
        for goal in self.maze.GOAL_STATES:
            ax.scatter([goal[1]], [goal[0]], [0.1], 
                      c='blue', s=500, marker='*', label='Goal' if goal == self.maze.GOAL_STATES[0] else '', zorder=10)
        
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_zlabel('Height', fontsize=12)
        ax.set_title('3D Maze Structure', fontsize=14, fontweight='bold')
        ax.set_xlim([-0.5, self.maze.WORLD_WIDTH - 0.5])
        ax.set_ylim([-0.5, self.maze.WORLD_HEIGHT - 0.5])
        ax.set_zlim([-0.1, 1.5])
        ax.legend()
        ax.view_init(elev=60, azim=45)
        
        plt.savefig(os.path.join(IMAGE_DIR, 'maze_3d_structure.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def simulate_path(self, q_value, epsilon=0.1, max_steps=1000):
        """Simulate a path through the maze"""
        state = self.maze.START_STATE.copy()
        path = [state.copy()]
        total_reward = 0
        
        for step in range(max_steps):
            if state in self.maze.GOAL_STATES:
                break
                
            # Choose action
            if np.random.rand() < epsilon:
                action = np.random.choice(self.maze.actions)
            else:
                values = q_value[state[0], state[1], :]
                action = np.random.choice([a for a, v in enumerate(values) if v == np.max(values)])
            
            # Take step
            next_state, reward = self.maze.step(state, action)
            path.append(next_state.copy())
            total_reward += reward
            state = next_state
            
        return path, total_reward
        
    def visualize_path_3d(self, path, title="Agent Path"):
        """Visualize agent's path in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create maze surface
        X, Y = np.meshgrid(range(self.maze.WORLD_WIDTH), range(self.maze.WORLD_HEIGHT))
        Z = np.zeros_like(X)
        
        obstacle_mask = np.zeros_like(Z, dtype=bool)
        for obs in self.maze.obstacles:
            obstacle_mask[obs[0], obs[1]] = True
        
        colors = np.ones_like(Z)
        colors[obstacle_mask] = 0.3
        
        ax.plot_surface(X, Y, Z, facecolors=plt.cm.RdYlGn(colors), 
                       alpha=0.5, linewidth=0.5, antialiased=True)
        
        # Draw obstacles
        for obs in self.maze.obstacles:
            ax.bar3d(obs[1], obs[0], 0, 0.8, 0.8, 1.0, color='red', alpha=0.6)
        
        # Extract path coordinates
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            path_z = [0.1 + i * 0.02 for i in range(len(path))]
            
            # Plot path
            ax.plot(path_x, path_y, path_z, 'b-', linewidth=3, label='Agent Path', zorder=5)
            ax.scatter(path_x, path_y, path_z, c=range(len(path)), 
                      cmap='cool', s=100, alpha=0.8, zorder=6)
            
            # Mark start and end
            ax.scatter([path_x[0]], [path_y[0]], [path_z[0]], 
                      c='green', s=500, marker='o', label='Start', zorder=10)
            ax.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], 
                      c='red', s=500, marker='*', label='End', zorder=10)
        
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_zlabel('Time Step', fontsize=12)
        ax.set_title(f'3D {title}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.view_init(elev=60, azim=45)
        
        plt.savefig(os.path.join(IMAGE_DIR, f'maze_3d_{title.lower().replace(" ", "_")}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_q_values_3d(self, q_value):
        """Visualize Q-values as 3D surfaces for each action"""
        fig = plt.figure(figsize=(18, 12))
        
        action_names = ['Up', 'Down', 'Left', 'Right']
        
        for action_idx, action_name in enumerate(action_names):
            ax = fig.add_subplot(2, 2, action_idx + 1, projection='3d')
            
            # Extract Q-values for this action
            Q_action = q_value[:, :, action_idx]
            
            # Create meshgrid
            X, Y = np.meshgrid(range(self.maze.WORLD_WIDTH), range(self.maze.WORLD_HEIGHT))
            Z = Q_action
            
            # Plot surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                                 linewidth=0.5, antialiased=True, edgecolor='black')
            
            # Mark special positions
            ax.scatter([self.maze.START_STATE[1]], [self.maze.START_STATE[0]], 
                      [Q_action[self.maze.START_STATE[0], self.maze.START_STATE[1]]], 
                      c='green', s=200, marker='o', zorder=10)
            
            for goal in self.maze.GOAL_STATES:
                ax.scatter([goal[1]], [goal[0]], 
                          [Q_action[goal[0], goal[1]]], 
                          c='red', s=200, marker='*', zorder=10)
            
            ax.set_xlabel('Column', fontsize=10)
            ax.set_ylabel('Row', fontsize=10)
            ax.set_zlabel('Q-Value', fontsize=10)
            ax.set_title(f'Q-Values for Action: {action_name}', fontsize=12, fontweight='bold')
            ax.view_init(elev=45, azim=45)
            
            plt.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'maze_3d_q_values.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main function to demonstrate 3D visualizations"""
    maze_3d = Maze3D()
    
    print("Visualizing maze structure...")
    maze_3d.visualize_maze_3d()
    
    # Create sample Q-values
    print("\nCreating sample Q-values...")
    q_value = np.random.randn(maze_3d.maze.WORLD_HEIGHT, 
                              maze_3d.maze.WORLD_WIDTH, 
                              len(maze_3d.maze.actions)) * 0.1
    
    # Set goal to have high values
    for goal in maze_3d.maze.GOAL_STATES:
        q_value[goal[0], goal[1], :] = 10
    
    print("\nVisualizing Q-values...")
    maze_3d.visualize_q_values_3d(q_value)
    
    # Simulate a path
    print("\nSimulating path...")
    path, reward = maze_3d.simulate_path(q_value, epsilon=0.1)
    print(f"Path completed with {len(path)} steps, total reward: {reward}")
    
    print("\nVisualizing path...")
    maze_3d.visualize_path_3d(path, "Sample Agent Path")


if __name__ == '__main__':
    main()
