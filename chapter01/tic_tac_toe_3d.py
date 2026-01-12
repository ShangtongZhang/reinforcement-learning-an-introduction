"""
3D Interactive Tic-Tac-Toe Game
Visualizes the game board in 3D with interactive gameplay
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import pickle
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter01.tic_tac_toe import State, Player, Judger, HumanPlayer, all_states

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class TicTacToe3D:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.current_state = State()
        self.player1 = None
        self.player2 = None
        self.judger = None
        
    def draw_board_3d(self, state):
        """Draw the 3D tic-tac-toe board"""
        self.ax.clear()
        
        # Board dimensions
        board_size = 3
        spacing = 1.2
        
        # Draw grid lines
        for i in range(board_size + 1):
            # Vertical lines
            self.ax.plot([i, i], [0, board_size], [0, 0], 'k-', linewidth=2)
            self.ax.plot([i, i], [0, board_size], [0.1, 0.1], 'k-', linewidth=2)
            # Horizontal lines
            self.ax.plot([0, board_size], [i, i], [0, 0], 'k-', linewidth=2)
            self.ax.plot([0, board_size], [i, i], [0.1, 0.1], 'k-', linewidth=2)
            # Vertical connectors
            self.ax.plot([i, i], [0, 0], [0, 0.1], 'k-', linewidth=2)
            self.ax.plot([i, i], [board_size, board_size], [0, 0.1], 'k-', linewidth=2)
        
        # Draw pieces
        for i in range(board_size):
            for j in range(board_size):
                x, y = j + 0.5, i + 0.5
                z = 0.05
                
                if state.data[i, j] == 1:  # Player 1 (X)
                    # Draw X as two crossing lines in 3D
                    size = 0.3
                    self.ax.plot([x-size, x+size], [y-size, y+size], [z, z], 'r-', linewidth=4)
                    self.ax.plot([x-size, x+size], [y+size, y-size], [z, z], 'r-', linewidth=4)
                    # Add depth
                    self.ax.plot([x-size, x-size], [y-size, y-size], [z, z+0.05], 'r-', linewidth=2, alpha=0.5)
                    self.ax.plot([x+size, x+size], [y+size, y+size], [z, z+0.05], 'r-', linewidth=2, alpha=0.5)
                    
                elif state.data[i, j] == -1:  # Player 2 (O)
                    # Draw O as a circle in 3D
                    theta = np.linspace(0, 2*np.pi, 50)
                    radius = 0.3
                    circle_x = x + radius * np.cos(theta)
                    circle_y = y + radius * np.sin(theta)
                    circle_z = np.full_like(theta, z)
                    self.ax.plot(circle_x, circle_y, circle_z, 'b-', linewidth=4)
                    # Add depth ring
                    self.ax.plot(circle_x, circle_y, circle_z + 0.05, 'b-', linewidth=2, alpha=0.5)
        
        # Set labels and title
        self.ax.set_xlabel('Column', fontsize=12)
        self.ax.set_ylabel('Row', fontsize=12)
        self.ax.set_zlabel('Height', fontsize=12)
        self.ax.set_title('3D Tic-Tac-Toe Board', fontsize=14, fontweight='bold')
        
        # Set limits
        self.ax.set_xlim([-0.5, board_size + 0.5])
        self.ax.set_ylim([-0.5, board_size + 0.5])
        self.ax.set_zlim([-0.1, 0.3])
        
        # Set viewing angle
        self.ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        
    def visualize_game(self, state):
        """Visualize a single game state"""
        self.draw_board_3d(state)
        plt.savefig(os.path.join(IMAGE_DIR, 'tic_tac_toe_3d.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_value_function(self):
        """Visualize the learned value function in 3D"""
        player = Player(epsilon=0)
        try:
            player.load_policy()
        except:
            print("No saved policy found. Please train first.")
            return
            
        # Create 3D surface of value function
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get all non-terminal states
        states_3d = []
        values = []
        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            if not is_end:
                # Convert state to 3D coordinates (row, col, value)
                # Find first empty position as representative
                for i in range(3):
                    for j in range(3):
                        if state.data[i, j] == 0:
                            states_3d.append([i, j, hash_val % 100])  # Use hash as z
                            values.append(player.estimations.get(hash_val, 0.5))
                            break
                    else:
                        continue
                    break
        
        if states_3d:
            states_3d = np.array(states_3d)
            values = np.array(values)
            
            # Create surface
            ax.scatter(states_3d[:, 0], states_3d[:, 1], states_3d[:, 2], 
                      c=values, cmap='RdYlGn', s=50, alpha=0.7)
            
            ax.set_xlabel('Row', fontsize=12)
            ax.set_ylabel('Column', fontsize=12)
            ax.set_zlabel('State Hash', fontsize=12)
            ax.set_title('3D Value Function Visualization', fontsize=14, fontweight='bold')
            
            plt.colorbar(ax.scatter(states_3d[:, 0], states_3d[:, 1], states_3d[:, 2], 
                                   c=values, cmap='RdYlGn', s=50, alpha=0.7), ax=ax)
            
            plt.savefig(os.path.join(IMAGE_DIR, 'tic_tac_toe_value_3d.png'), dpi=150, bbox_inches='tight')
            plt.show()


def main():
    """Main function to demonstrate 3D visualizations"""
    visualizer = TicTacToe3D()
    
    # Create a sample game state
    state = State()
    state.data[0, 0] = 1   # X
    state.data[0, 1] = -1  # O
    state.data[1, 1] = 1   # X
    state.data[1, 2] = -1  # O
    
    print("Visualizing sample game state...")
    visualizer.visualize_game(state)
    
    # Try to visualize value function if policy exists
    print("\nAttempting to visualize value function...")
    visualizer.visualize_value_function()


if __name__ == '__main__':
    main()
