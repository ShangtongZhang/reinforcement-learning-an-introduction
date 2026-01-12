"""
3D Visualization of Policy Gradient Methods (Chapter 13)
Shows policy parameter space and gradient ascent
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class PolicyGradient3D:
    def visualize_policy_surface_3d(self):
        """Visualize policy parameter space"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Policy parameters θ (simplified 2D parameter space)
        theta1 = np.linspace(-2, 2, 50)
        theta2 = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(theta1, theta2)
        
        # Expected return surface (simplified)
        # True value function for short corridor: v(s) = (2p - 4) / (p(1-p))
        # where p = probability of going right
        Z = np.zeros_like(X)
        for i in range(len(theta1)):
            for j in range(len(theta2)):
                # Softmax to get probability
                h = np.array([theta1[i], theta2[j]])
                exp_h = np.exp(h - np.max(h))
                prob = exp_h / np.sum(exp_h)
                p_right = prob[1]
                
                # Value function (avoiding division by zero)
                if p_right > 0.01 and p_right < 0.99:
                    Z[i, j] = (2 * p_right - 4) / (p_right * (1 - p_right))
                else:
                    Z[i, j] = -100
        
        # Clip extreme values for visualization
        Z = np.clip(Z, -50, 10)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9,
                               linewidth=0.5, antialiased=True)
        
        # Mark optimal point
        optimal_idx = np.unravel_index(np.argmax(Z), Z.shape)
        ax.scatter([X[optimal_idx]], [Y[optimal_idx]], [Z[optimal_idx]], 
                  c='red', s=300, marker='*', label='Optimal Policy', zorder=10)
        
        ax.set_xlabel('θ₁ (Parameter 1)', fontsize=12)
        ax.set_ylabel('θ₂ (Parameter 2)', fontsize=12)
        ax.set_zlabel('Expected Return', fontsize=12)
        ax.set_title('3D Policy Parameter Space', fontsize=14, fontweight='bold')
        ax.legend()
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'policy_gradient_3d_surface.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_gradient_ascent_3d(self):
        """Visualize gradient ascent trajectory in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create performance surface
        theta1 = np.linspace(-2, 2, 40)
        theta2 = np.linspace(-2, 2, 40)
        X, Y = np.meshgrid(theta1, theta2)
        Z = -(X**2 + Y**2) + 10  # Simple quadratic (concave)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6,
                               linewidth=0.5, antialiased=True)
        
        # Simulate gradient ascent trajectory
        theta = np.array([-1.5, -1.5])
        alpha = 0.1
        trajectory = [theta.copy()]
        
        for step in range(50):
            # Gradient (simplified)
            grad = -2 * theta
            theta = theta + alpha * grad
            trajectory.append(theta.copy())
        
        trajectory = np.array(trajectory)
        
        # Plot trajectory
        traj_z = -(trajectory[:, 0]**2 + trajectory[:, 1]**2) + 10
        ax.plot(trajectory[:, 0], trajectory[:, 1], traj_z, 
               'r-', linewidth=3, label='Gradient Ascent Path', zorder=10)
        ax.scatter(trajectory[:, 0], trajectory[:, 1], traj_z,
                  c=range(len(trajectory)), cmap='cool', s=100, zorder=11)
        
        # Mark start and end
        ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], [traj_z[0]],
                  c='green', s=300, marker='o', label='Start', zorder=12)
        ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], [traj_z[-1]],
                  c='red', s=300, marker='*', label='Optimum', zorder=12)
        
        ax.set_xlabel('θ₁', fontsize=12)
        ax.set_ylabel('θ₂', fontsize=12)
        ax.set_zlabel('Expected Return', fontsize=12)
        ax.set_title('3D REINFORCE: Gradient Ascent Trajectory', fontsize=14, fontweight='bold')
        ax.legend()
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'policy_gradient_3d_ascent.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


def main():
    pg3d = PolicyGradient3D()
    print("Visualizing policy parameter space...")
    pg3d.visualize_policy_surface_3d()
    print("\nVisualizing gradient ascent...")
    pg3d.visualize_gradient_ascent_3d()
    print("Complete!")


if __name__ == '__main__':
    main()
