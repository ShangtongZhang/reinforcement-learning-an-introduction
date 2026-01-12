"""
3D Visualization of Baird's Counterexample (Chapter 11)
Shows weight divergence in off-policy learning
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class Counterexample3D:
    def visualize_weight_divergence_3d(self):
        """Visualize weight divergence over time in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Simulate weight updates (simplified model of divergence)
        n_steps = 1000
        n_weights = 8
        
        weights_history = np.zeros((n_steps, n_weights))
        weights = np.ones(n_weights)
        weights[6] = 10  # Initial condition
        
        for step in range(n_steps):
            weights_history[step] = weights
            # Simplified divergence model
            weights = weights * 1.01 + np.random.randn(n_weights) * 0.1
        
        # Plot weight trajectories
        for w_idx in range(n_weights):
            X = np.arange(n_steps)
            Y = np.full_like(X, w_idx)
            Z = weights_history[:, w_idx]
            
            ax.plot(X, Y, Z, linewidth=2, alpha=0.8,
                   label=f'θ{w_idx+1}', color=plt.cm.viridis(w_idx / n_weights))
        
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Weight Index', fontsize=12)
        ax.set_zlabel('Weight Value', fontsize=12)
        ax.set_title("3D Baird's Counterexample: Weight Divergence", fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.view_init(elev=30, azim=45)
        
        plt.savefig(os.path.join(IMAGE_DIR, 'counterexample_3d_divergence.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_tdc_convergence_3d(self):
        """Visualize TDC algorithm convergence vs divergence"""
        fig = plt.figure(figsize=(16, 12))
        
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        n_steps = 500
        n_weights = 8
        
        # Semi-gradient TD (diverges)
        weights_div = np.ones(n_weights)
        weights_div[6] = 10
        history_div = np.zeros((n_steps, n_weights))
        
        for step in range(n_steps):
            history_div[step] = weights_div
            weights_div = weights_div * 1.02 + np.random.randn(n_weights) * 0.05
        
        # TDC (converges)
        weights_conv = np.ones(n_weights)
        weights_conv[6] = 10
        history_conv = np.zeros((n_steps, n_weights))
        target = np.zeros(n_weights)
        
        for step in range(n_steps):
            history_conv[step] = weights_conv
            weights_conv = weights_conv * 0.99 + (target - weights_conv) * 0.01
        
        # Plot divergence
        for w_idx in range(n_weights):
            ax1.plot(np.arange(n_steps), np.full(n_steps, w_idx), history_div[:, w_idx],
                    linewidth=2, alpha=0.7, color=plt.cm.Reds(w_idx / n_weights))
        
        ax1.set_xlabel('Steps', fontsize=10)
        ax1.set_ylabel('Weight', fontsize=10)
        ax1.set_zlabel('Value', fontsize=10)
        ax1.set_title('Semi-gradient TD (Diverges)', fontsize=12, fontweight='bold')
        ax1.view_init(elev=30, azim=45)
        
        # Plot convergence
        for w_idx in range(n_weights):
            ax2.plot(np.arange(n_steps), np.full(n_steps, w_idx), history_conv[:, w_idx],
                    linewidth=2, alpha=0.7, color=plt.cm.Greens(w_idx / n_weights))
        
        ax2.set_xlabel('Steps', fontsize=10)
        ax2.set_ylabel('Weight', fontsize=10)
        ax2.set_zlabel('Value', fontsize=10)
        ax2.set_title('TDC (Converges)', fontsize=12, fontweight='bold')
        ax2.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'counterexample_3d_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


def main():
    ce3d = Counterexample3D()
    print("Visualizing Baird's Counterexample...")
    ce3d.visualize_weight_divergence_3d()
    print("\nVisualizing TDC vs Semi-gradient TD...")
    ce3d.visualize_tdc_convergence_3d()
    print("Complete!")


if __name__ == '__main__':
    main()
