"""
Interactive 3D Cliff Walking Game
Control the agent with arrow keys to navigate the cliff
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter06.cliff_walking import step, ACTIONS, START, GOAL, WORLD_HEIGHT, WORLD_WIDTH

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class CliffWalking3DGame:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.world_height = WORLD_HEIGHT
        self.world_width = WORLD_WIDTH
        self.state = START.copy()
        self.path = [START.copy()]
        self.total_reward = 0
        self.steps = 0
        self.game_over = False
        self.auto_mode = False
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Create control buttons
        ax_reset = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_game)
        
        ax_auto = plt.axes([0.81, 0.05, 0.1, 0.04])
        self.btn_auto = Button(ax_auto, 'Auto Play')
        self.btn_auto.on_clicked(self.toggle_auto)
        
    def draw_environment_3d(self):
        """Draw the 3D cliff walking environment"""
        self.ax.clear()
        
        # Create grid
        X, Y = np.meshgrid(range(self.world_width), range(self.world_height))
        Z = np.zeros_like(X)
        
        # Mark cliff area (row 2, columns 1-10)
        cliff_mask = np.zeros_like(Z, dtype=bool)
        cliff_mask[2, 1:11] = True
        
        # Create surface with different colors
        colors = np.ones_like(Z)
        colors[cliff_mask] = 0.3
        
        # Plot surface
        self.ax.plot_surface(X, Y, Z, facecolors=plt.cm.RdYlGn(colors), 
                       alpha=0.7, linewidth=0.5, antialiased=True)
        
        # Mark cliff with red bars
        for j in range(1, 11):
            self.ax.bar3d(j, 2, 0, 0.8, 0.8, -0.5, color='red', alpha=0.8)
        
        # Mark start position
        self.ax.scatter([START[1]], [START[0]], [0.1], 
                  c='green', s=500, marker='o', label='Start', zorder=10)
        
        # Mark goal position
        self.ax.scatter([GOAL[1]], [GOAL[0]], [0.1], 
                  c='blue', s=500, marker='*', label='Goal', zorder=10)
        
        # Draw path
        if len(self.path) > 1:
            path_x = [p[1] for p in self.path]
            path_y = [p[0] for p in self.path]
            path_z = [0.1 + i * 0.01 for i in range(len(self.path))]
            
            self.ax.plot(path_x, path_y, path_z, 'b-', linewidth=3, label='Path', zorder=5)
            self.ax.scatter(path_x, path_y, path_z, c=range(len(self.path)), 
                      cmap='cool', s=100, alpha=0.8, zorder=6)
        
        # Mark current position
        self.ax.scatter([self.state[1]], [self.state[0]], [0.2], 
                  c='yellow', s=600, marker='D', label='Current', zorder=10, edgecolors='black', linewidths=2)
        
        # Add status text
        status = f"Steps: {self.steps} | Reward: {self.total_reward} | Position: ({self.state[0]}, {self.state[1]})"
        if self.game_over:
            if self.state == GOAL:
                status += " | Goal Reached!"
            else:
                status += " | Fell off cliff!"
        
        self.ax.text2D(0.05, 0.95, status, transform=self.ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.ax.set_xlabel('Column', fontsize=12)
        self.ax.set_ylabel('Row', fontsize=12)
        self.ax.set_zlabel('Height', fontsize=12)
        self.ax.set_title('3D Cliff Walking - Use Arrow Keys to Move!', fontsize=14, fontweight='bold')
        self.ax.set_xlim([-0.5, WORLD_WIDTH - 0.5])
        self.ax.set_ylim([-0.5, WORLD_HEIGHT - 0.5])
        self.ax.set_zlim([-0.6, 0.5])
        self.ax.legend(loc='upper right')
        self.ax.view_init(elev=60, azim=45)
        
        plt.tight_layout()
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if self.game_over:
            return
        
        action_map = {
            'up': ACTIONS[0],      # ACTION_UP
            'down': ACTIONS[1],    # ACTION_DOWN
            'left': ACTIONS[2],    # ACTION_LEFT
            'right': ACTIONS[3],   # ACTION_RIGHT
        }
        
        if event.key in action_map:
            self.take_action(action_map[event.key])
        elif event.key == 'r':
            self.reset_game()
    
    def take_action(self, action):
        """Take an action and update the game state"""
        next_state, reward = step(self.state, action)
        
        self.state = next_state
        self.path.append(self.state.copy())
        self.total_reward += reward
        self.steps += 1
        
        # Check if game is over
        if self.state == GOAL:
            self.game_over = True
            print(f"Goal reached! Total steps: {self.steps}, Total reward: {self.total_reward}")
        elif reward == -100:  # Fell off cliff
            self.game_over = True
            print(f"Fell off cliff! Total steps: {self.steps}, Total reward: {self.total_reward}")
        
        self.draw_environment_3d()
        plt.draw()
    
    def reset_game(self, event=None):
        """Reset the game"""
        self.state = START.copy()
        self.path = [START.copy()]
        self.total_reward = 0
        self.steps = 0
        self.game_over = False
        self.draw_environment_3d()
        plt.draw()
        print("Game reset!")
    
    def toggle_auto(self, event=None):
        """Toggle auto-play mode"""
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            print("Auto-play enabled. Press 'r' to reset.")
            self.auto_play()
        else:
            print("Auto-play disabled.")
    
    def auto_play(self):
        """Auto-play using random actions"""
        import time
        while self.auto_mode and not self.game_over:
            action = np.random.choice(ACTIONS)
            self.take_action(action)
            plt.pause(0.3)
            if self.game_over:
                self.auto_mode = False
    
    def run(self):
        """Run the game"""
        self.draw_environment_3d()
        print("\nControls:")
        print("  Arrow Keys: Move agent")
        print("  R: Reset game")
        print("  Auto Play button: Enable auto-play")
        plt.show()


def main():
    """Main function"""
    game = CliffWalking3DGame()
    game.run()


if __name__ == '__main__':
    main()
