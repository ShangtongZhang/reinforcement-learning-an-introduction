"""
Interactive 3D Mountain Car Game
Control the car with keyboard to reach the goal
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter10.mountain_car import step, ACTIONS, POSITION_MIN, POSITION_MAX, VELOCITY_MIN, VELOCITY_MAX

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class MountainCar3DInteractive:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.position = np.random.uniform(-0.6, -0.4)
        self.velocity = 0.0
        self.steps = 0
        self.max_steps = 1000
        self.trajectory = [(self.position, self.velocity)]
        self.game_over = False
        self.auto_mode = False
        self.current_action = ACTIONS[1]  # Start with no action
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        
        # Create control buttons
        ax_reset = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_game)
        
        ax_auto = plt.axes([0.81, 0.05, 0.1, 0.04])
        self.btn_auto = Button(ax_auto, 'Auto Play')
        self.btn_auto.on_clicked(self.toggle_auto)
        
        # Animation
        self.animation = None
        
    def draw_environment_3d(self):
        """Draw the 3D mountain car environment"""
        self.ax.clear()
        
        # Create mountain landscape
        x = np.linspace(POSITION_MIN, POSITION_MAX, 100)
        y_mountain = np.sin(3 * x)
        
        # Create 3D surface
        X = np.linspace(POSITION_MIN, POSITION_MAX, 50)
        Y = np.linspace(VELOCITY_MIN, VELOCITY_MAX, 50)
        X, Y = np.meshgrid(X, Y)
        Z = np.sin(3 * X)  # Mountain surface
        
        # Plot mountain surface
        self.ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6, 
                             linewidth=0.5, antialiased=True)
        
        # Mark goal position
        goal_x = POSITION_MAX
        goal_y = 0
        goal_z = np.sin(3 * goal_x)
        self.ax.scatter([goal_x], [goal_y], [goal_z], 
                  c='green', s=500, marker='*', label='Goal', zorder=10)
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            traj_x = [t[0] for t in self.trajectory]
            traj_y = [t[1] for t in self.trajectory]
            traj_z = [np.sin(3 * x) for x in traj_x]
            
            self.ax.plot(traj_x, traj_y, traj_z, 'b-', linewidth=2, 
                        label='Trajectory', zorder=5, alpha=0.7)
        
        # Mark current car position
        car_x = self.position
        car_y = self.velocity
        car_z = np.sin(3 * car_x)
        self.ax.scatter([car_x], [car_y], [car_z], 
                  c='red', s=600, marker='o', label='Car', zorder=10, 
                  edgecolors='black', linewidths=2)
        
        # Add status text
        status = (f"Position: {self.position:.3f} | Velocity: {self.velocity:.3f} | "
                 f"Steps: {self.steps}/{self.max_steps}")
        if self.game_over:
            if self.position >= POSITION_MAX:
                status += " | Goal Reached!"
            else:
                status += " | Time Limit Reached"
        
        action_text = "Action: "
        if self.current_action == ACTIONS[0]:
            action_text += "Reverse (Left Arrow)"
        elif self.current_action == ACTIONS[1]:
            action_text += "Neutral (No Key)"
        else:
            action_text += "Forward (Right Arrow)"
        
        self.ax.text2D(0.05, 0.95, status, transform=self.ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.ax.text2D(0.05, 0.88, action_text, transform=self.ax.transAxes, 
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        self.ax.set_xlabel('Position', fontsize=12)
        self.ax.set_ylabel('Velocity', fontsize=12)
        self.ax.set_zlabel('Height', fontsize=12)
        self.ax.set_title('3D Mountain Car - Use Arrow Keys to Control!', fontsize=14, fontweight='bold')
        self.ax.set_xlim([POSITION_MIN - 0.1, POSITION_MAX + 0.1])
        self.ax.set_ylim([VELOCITY_MIN - 0.01, VELOCITY_MAX + 0.01])
        self.ax.set_zlim([-1.5, 1.5])
        self.ax.legend(loc='upper right')
        self.ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if self.game_over:
            return
        
        if event.key == 'left':
            self.current_action = ACTIONS[0]  # Reverse
        elif event.key == 'right':
            self.current_action = ACTIONS[2]  # Forward
        elif event.key == 'r':
            self.reset_game()
    
    def on_key_release(self, event):
        """Handle key release"""
        if event.key in ['left', 'right']:
            self.current_action = ACTIONS[1]  # Neutral
    
    def update(self, frame):
        """Update function for animation"""
        if not self.game_over and not self.auto_mode:
            # Take action based on current key state
            reward = self.take_action(self.current_action)
        elif self.auto_mode and not self.game_over:
            # Random action for auto mode
            action = np.random.choice(ACTIONS)
            self.take_action(action)
        
        self.draw_environment_3d()
        return []
    
    def take_action(self, action):
        """Take an action and update the game state"""
        self.position, self.velocity, reward = step(self.position, self.velocity, action)
        self.steps += 1
        self.trajectory.append((self.position, self.velocity))
        
        # Keep trajectory to last 100 points for performance
        if len(self.trajectory) > 100:
            self.trajectory = self.trajectory[-100:]
        
        # Check if game is over
        if self.position >= POSITION_MAX:
            self.game_over = True
            print(f"Goal reached! Steps: {self.steps}")
        elif self.steps >= self.max_steps:
            self.game_over = True
            print(f"Time limit reached. Final position: {self.position:.3f}")
        
        return reward
    
    def reset_game(self, event=None):
        """Reset the game"""
        self.position = np.random.uniform(-0.6, -0.4)
        self.velocity = 0.0
        self.steps = 0
        self.trajectory = [(self.position, self.velocity)]
        self.game_over = False
        self.current_action = ACTIONS[1]
        self.draw_environment_3d()
        plt.draw()
        print("Game reset!")
    
    def toggle_auto(self, event=None):
        """Toggle auto-play mode"""
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            print("Auto-play enabled. Press 'r' to reset.")
        else:
            print("Auto-play disabled.")
    
    def run(self):
        """Run the game"""
        self.draw_environment_3d()
        print("\nControls:")
        print("  Left Arrow: Reverse (push left)")
        print("  Right Arrow: Forward (push right)")
        print("  Release: Neutral (coast)")
        print("  R: Reset game")
        print("  Auto Play button: Enable auto-play")
        print("\nGoal: Reach the top of the mountain (position >= 0.5)")
        
        # Start animation
        self.animation = FuncAnimation(self.fig, self.update, interval=50, blit=False)
        plt.show()


def main():
    """Main function"""
    game = MountainCar3DInteractive()
    game.run()


if __name__ == '__main__':
    main()
