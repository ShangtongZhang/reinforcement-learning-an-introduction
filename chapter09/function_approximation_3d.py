"""
3D Visualization of Function Approximation (Chapter 9)
Shows tile coding, fourier basis, and polynomial approximation surfaces
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class FunctionApproximation3D:
    def visualize_basis_functions_3d(self):
        """Visualize different basis functions in 3D"""
        fig = plt.figure(figsize=(18, 12))
        
        x = np.linspace(0, 1, 100)
        
        # Polynomial basis
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        for order in range(1, 6):
            y = x ** order
            X = x
            Y = np.full_like(X, order)
            Z = y
            ax1.plot(X, Y, Z, linewidth=3, label=f'x^{order}')
        
        ax1.set_xlabel('State (x)', fontsize=10)
        ax1.set_ylabel('Polynomial Order', fontsize=10)
        ax1.set_zlabel('Basis Value', fontsize=10)
        ax1.set_title('Polynomial Basis Functions', fontsize=12, fontweight='bold')
        ax1.legend()
        
        # Fourier basis
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        for order in range(1, 6):
            y = np.cos(order * np.pi * x)
            X = x
            Y = np.full_like(X, order)
            Z = y
            ax2.plot(X, Y, Z, linewidth=3, label=f'cos({order}πx)')
        
        ax2.set_xlabel('State (x)', fontsize=10)
        ax2.set_ylabel('Fourier Order', fontsize=10)
        ax2.set_zlabel('Basis Value', fontsize=10)
        ax2.set_title('Fourier Basis Functions', fontsize=12, fontweight='bold')
        ax2.legend()
        
        # Tile coding visualization
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        num_tilings = 5
        tile_width = 0.2
        
        for tiling in range(num_tilings):
            offset = tiling * 0.04
            X = x
            Y = np.full_like(X, tiling)
            Z = np.floor((X + offset) / tile_width)
            ax3.plot(X, Y, Z, linewidth=3, alpha=0.8)
        
        ax3.set_xlabel('State (x)', fontsize=10)
        ax3.set_ylabel('Tiling Index', fontsize=10)
        ax3.set_zlabel('Active Tile', fontsize=10)
        ax3.set_title('Tile Coding', fontsize=12, fontweight='bold')
        
        # Approximated function surface
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        
        # Create a target function
        true_function = np.sin(2 * np.pi * x) * x
        
        # Approximate with different methods
        poly_approx = x * (1 - x) * 2  # Simple polynomial
        fourier_approx = 0.5 * np.cos(np.pi * x) + 0.3 * np.cos(2 * np.pi * x)
        
        ax4.plot(x, np.zeros_like(x), true_function, 'k-', linewidth=3, label='True Function')
        ax4.plot(x, np.ones_like(x), poly_approx, 'b-', linewidth=2, label='Polynomial')
        ax4.plot(x, np.ones_like(x) * 2, fourier_approx, 'r-', linewidth=2, label='Fourier')
        
        ax4.set_xlabel('State', fontsize=10)
        ax4.set_ylabel('Method', fontsize=10)
        ax4.set_zlabel('Value', fontsize=10)
        ax4.set_title('Function Approximation Comparison', fontsize=12, fontweight='bold')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'function_approximation_3d.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


def main():
    fa3d = FunctionApproximation3D()
    print("Creating function approximation visualizations...")
    fa3d.visualize_basis_functions_3d()
    print("Complete!")


if __name__ == '__main__':
    main()
