"""
3D Visualization of Multi-Armed Bandits
Shows reward distributions and learning progress in 3D
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class Bandits3D:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.fig = plt.figure(figsize=(16, 12))
        
    def visualize_reward_distributions(self):
        """Visualize reward distributions for each arm in 3D"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate true action values (q*(a))
        np.random.seed(42)
        true_values = np.random.randn(self.n_arms)
        
        # Create positions for each arm
        x_pos = np.arange(self.n_arms)
        y_pos = np.zeros(self.n_arms)
        
        # Sample rewards for visualization
        n_samples = 200
        rewards = []
        for i in range(self.n_arms):
            arm_rewards = np.random.randn(n_samples) + true_values[i]
            rewards.append(arm_rewards)
        
        # Create violin plot-like visualization in 3D
        for i in range(self.n_arms):
            # Create histogram data
            hist, bins = np.histogram(rewards[i], bins=20)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Normalize histogram for width
            max_hist = hist.max()
            if max_hist > 0:
                widths = hist / max_hist * 0.4
            else:
                widths = np.zeros_like(hist)
            
            # Draw 3D bars
            for j, (center, width, height) in enumerate(zip(bin_centers, widths, hist)):
                if height > 0:
                    # Create bar vertices
                    x = [i - width, i + width, i + width, i - width]
                    y = [center - 0.1, center - 0.1, center + 0.1, center + 0.1]
                    z = [0, 0, height, height]
                    
                    # Draw bar
                    verts = [list(zip(x, y, z))]
                    ax.add_collection3d(Poly3DCollection(verts, alpha=0.6, 
                                                       facecolor=plt.cm.viridis(i/self.n_arms),
                                                       edgecolor='black', linewidth=0.5))
            
            # Mark true value
            ax.scatter([i], [true_values[i]], [0], c='red', s=100, marker='*', 
                      label='True Value' if i == 0 else '', zorder=10)
        
        ax.set_xlabel('Arm', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_zlabel('Frequency', fontsize=12)
        ax.set_title('3D Multi-Armed Bandit Reward Distributions', fontsize=14, fontweight='bold')
        ax.set_xticks(range(self.n_arms))
        ax.legend()
        ax.view_init(elev=20, azim=45)
        
        plt.savefig(os.path.join(IMAGE_DIR, 'bandits_3d_distributions.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_learning_progress(self, steps=1000, epsilon=0.1):
        """Visualize learning progress over time in 3D"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # True action values
        np.random.seed(42)
        true_values = np.random.randn(self.n_arms)
        best_action = np.argmax(true_values)
        
        # Initialize Q values
        Q = np.zeros(self.n_arms)
        N = np.zeros(self.n_arms)
        
        # Track learning
        time_steps = []
        action_chosen = []
        rewards_received = []
        q_values_history = []
        
        for t in range(steps):
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(self.n_arms)
            else:
                action = np.argmax(Q)
            
            # Sample reward
            reward = np.random.randn() + true_values[action]
            
            # Update Q value
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            
            # Store data
            time_steps.append(t)
            action_chosen.append(action)
            rewards_received.append(reward)
            q_values_history.append(Q.copy())
        
        # Convert to arrays
        time_steps = np.array(time_steps)
        action_chosen = np.array(action_chosen)
        rewards_received = np.array(rewards_received)
        q_values_history = np.array(q_values_history)
        
        # Create 3D surface plot of Q values over time
        X, Y = np.meshgrid(time_steps, range(self.n_arms))
        Z = q_values_history.T
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
        
        # Mark best action
        ax.scatter([0, steps-1], [best_action, best_action], 
                  [true_values[best_action], true_values[best_action]], 
                  c='red', s=100, marker='*', label='True Best Action', zorder=10)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Arm', fontsize=12)
        ax.set_zlabel('Q Value', fontsize=12)
        ax.set_title(f'3D Learning Progress (ε={epsilon})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'bandits_3d_learning.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_action_selection_landscape(self, steps=1000):
        """Visualize the action selection landscape in 3D"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # True action values
        np.random.seed(42)
        true_values = np.random.randn(self.n_arms)
        
        # Simulate multiple runs
        n_runs = 10
        selection_counts = np.zeros((n_runs, self.n_arms, steps))
        
        for run in range(n_runs):
            Q = np.zeros(self.n_arms)
            N = np.zeros(self.n_arms)
            
            for t in range(steps):
                # Epsilon-greedy
                if np.random.rand() < 0.1:
                    action = np.random.randint(self.n_arms)
                else:
                    action = np.argmax(Q)
                
                # Update
                reward = np.random.randn() + true_values[action]
                N[action] += 1
                Q[action] += (reward - Q[action]) / N[action]
                
                selection_counts[run, action, t] = 1
        
        # Average across runs
        avg_selections = selection_counts.mean(axis=0)
        
        # Create 3D bar plot
        x_pos = np.arange(self.n_arms)
        y_pos = np.arange(steps)
        X, Y = np.meshgrid(x_pos, y_pos)
        Z = avg_selections.T
        
        # Plot as surface
        ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, linewidth=0, antialiased=True)
        
        ax.set_xlabel('Arm', fontsize=12)
        ax.set_ylabel('Time Step', fontsize=12)
        ax.set_zlabel('Selection Probability', fontsize=12)
        ax.set_title('3D Action Selection Landscape', fontsize=14, fontweight='bold')
        ax.view_init(elev=30, azim=45)
        
        plt.savefig(os.path.join(IMAGE_DIR, 'bandits_3d_selection.png'), dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main function to demonstrate 3D visualizations"""
    bandits = Bandits3D(n_arms=10)
    
    print("Visualizing reward distributions...")
    bandits.visualize_reward_distributions()
    
    print("\nVisualizing learning progress...")
    bandits.visualize_learning_progress(steps=1000, epsilon=0.1)
    
    print("\nVisualizing action selection landscape...")
    bandits.visualize_action_selection_landscape(steps=500)


if __name__ == '__main__':
    main()
