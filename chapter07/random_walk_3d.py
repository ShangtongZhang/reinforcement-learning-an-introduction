"""
3D Visualization of Random Walk (Chapter 7)
Shows n-step TD trajectories and value functions in 3D
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

N_STATES = 19

class RandomWalk3D:
    def __init__(self, n_states=19):
        self.n_states = n_states
        self.states = np.arange(1, n_states + 1)
        
    def visualize_value_convergence_3d(self, value_history):
        """Visualize value function convergence over iterations in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        n_iterations = len(value_history)
        
        for idx, values in enumerate(value_history[::max(1, n_iterations//10)]):
            X = self.states
            Y = np.full_like(X, idx)
            Z = values[1:-1]
            
            # Plot line for this iteration
            ax.plot(X, Y, Z, alpha=0.7, linewidth=2, 
                   color=plt.cm.viridis(idx / len(value_history)))
        
        ax.set_xlabel('State', fontsize=12)
        ax.set_ylabel('Iteration', fontsize=12)
        ax.set_zlabel('Value Estimate', fontsize=12)
        ax.set_title('3D Value Function Convergence', fontsize=14, fontweight='bold')
        ax.view_init(elev=30, azim=45)
        
        plt.savefig(os.path.join(IMAGE_DIR, 'random_walk_3d_convergence.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_trajectory_3d(self, trajectories):
        """Visualize multiple random walk trajectories in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        for traj_idx, trajectory in enumerate(trajectories[:10]):  # Limit to 10 trajectories
            X = np.arange(len(trajectory))
            Y = np.full_like(X, traj_idx)
            Z = trajectory
            
            ax.plot(X, Y, Z, alpha=0.7, linewidth=2,
                   color=plt.cm.cool(traj_idx / len(trajectories)))
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Episode', fontsize=12)
        ax.set_zlabel('State Position', fontsize=12)
        ax.set_title('3D Random Walk Trajectories', fontsize=14, fontweight='bold')
        ax.view_init(elev=30, azim=45)
        
        plt.savefig(os.path.join(IMAGE_DIR, 'random_walk_3d_trajectories.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_n_step_comparison_3d(self):
        """Visualize n-step TD methods comparison in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Different n values
        n_values = [1, 2, 4, 8, 16]
        alpha_values = np.linspace(0, 1, 20)
        
        # Create sample error surface (would be computed from actual runs)
        for n_idx, n in enumerate(n_values):
            errors = []
            for alpha in alpha_values:
                # Simplified error model
                error = 0.3 + abs(alpha - 0.5) * 0.4 + (1 / (n + 1)) * 0.2
                errors.append(error)
            
            X = alpha_values
            Y = np.full_like(X, n)
            Z = errors
            
            ax.plot(X, Y, Z, linewidth=3, alpha=0.8,
                   label=f'n={n}', color=plt.cm.viridis(n_idx / len(n_values)))
        
        ax.set_xlabel('Alpha (Learning Rate)', fontsize=12)
        ax.set_ylabel('n (TD Steps)', fontsize=12)
        ax.set_zlabel('RMS Error', fontsize=12)
        ax.set_title('3D n-step TD Performance Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.view_init(elev=30, azim=45)
        
        plt.savefig(os.path.join(IMAGE_DIR, 'random_walk_3d_nstep_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main function"""
    rw3d = RandomWalk3D()
    
    # Generate sample data for visualization
    print("Generating sample value convergence...")
    value_history = []
    current_values = np.zeros(N_STATES + 2)
    true_values = np.linspace(-1, 1, N_STATES + 2)
    
    for _ in range(20):
        current_values += (true_values - current_values) * 0.3
        value_history.append(current_values.copy())
    
    rw3d.visualize_value_convergence_3d(value_history)
    
    # Generate sample trajectories
    print("\nGenerating sample trajectories...")
    trajectories = []
    for _ in range(10):
        traj = [N_STATES // 2]  # Start in middle
        for _ in range(100):
            traj.append(max(0, min(N_STATES, traj[-1] + np.random.choice([-1, 1]))))
            if traj[-1] == 0 or traj[-1] == N_STATES:
                break
        trajectories.append(traj)
    
    rw3d.visualize_trajectory_3d(trajectories)
    
    # N-step comparison
    print("\nCreating n-step comparison...")
    rw3d.visualize_n_step_comparison_3d()
    
    print("\nAll Chapter 7 visualizations created!")


if __name__ == '__main__':
    main()
