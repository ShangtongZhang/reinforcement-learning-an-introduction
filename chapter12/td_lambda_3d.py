"""
3D Visualization of TD(λ) (Chapter 12)
Shows eligibility traces and lambda-return surfaces
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class TDLambda3D:
    def visualize_eligibility_traces_3d(self):
        """Visualize eligibility traces over time in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        n_states = 19
        n_steps = 50
        lambda_val = 0.9
        gamma = 1.0
        
        # Simulate eligibility traces
        traces = np.zeros((n_steps, n_states))
        
        # Agent visits states over time
        visited_states = [9]  # Start in middle
        for step in range(n_steps):
            # Decay all traces
            if step > 0:
                traces[step] = traces[step-1] * gamma * lambda_val
            
            # Visit a random nearby state
            if step < n_steps - 1:
                current = visited_states[-1]
                next_state = max(0, min(n_states-1, current + np.random.choice([-1, 1])))
                visited_states.append(next_state)
                traces[step][next_state] += 1  # Increment trace for visited state
        
        # Plot 3D surface
        X, Y = np.meshgrid(range(n_states), range(n_steps))
        Z = traces
        
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8,
                               linewidth=0.5, antialiased=True)
        
        ax.set_xlabel('State', fontsize=12)
        ax.set_ylabel('Time Step', fontsize=12)
        ax.set_zlabel('Eligibility Trace', fontsize=12)
        ax.set_title(f'3D Eligibility Traces (λ={lambda_val})', fontsize=14, fontweight='bold')
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'td_lambda_3d_traces.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_lambda_effect_3d(self):
        """Visualize effect of different lambda values in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        lambda_values = np.linspace(0, 1, 20)
        alpha_values = np.linspace(0, 1, 20)
        
        # Create performance surface (simplified model)
        X, Y = np.meshgrid(alpha_values, lambda_values)
        Z = np.zeros_like(X)
        
        for i, lam in enumerate(lambda_values):
            for j, alpha in enumerate(alpha_values):
                # Simplified error model: optimal around α=0.5, λ=0.9
                error = abs(alpha - 0.5) + abs(lam - 0.9) * 0.5
                Z[i, j] = error
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9,
                               linewidth=0.5, antialiased=True)
        
        # Mark optimal point
        ax.scatter([0.5], [0.9], [Z.min()], c='red', s=200, marker='*',
                  label='Optimal', zorder=10)
        
        ax.set_xlabel('Learning Rate (α)', fontsize=12)
        ax.set_ylabel('Lambda (λ)', fontsize=12)
        ax.set_zlabel('RMS Error', fontsize=12)
        ax.set_title('3D Parameter Space for TD(λ)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'td_lambda_3d_parameter_space.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


def main():
    td3d = TDLambda3D()
    print("Visualizing eligibility traces...")
    td3d.visualize_eligibility_traces_3d()
    print("\nVisualizing lambda parameter effect...")
    td3d.visualize_lambda_effect_3d()
    print("Complete!")


if __name__ == '__main__':
    main()
