"""
3D Visualization of Blackjack State Space
Shows value functions and state distributions in 3D
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class Blackjack3D:
    def __init__(self):
        # Blackjack state space dimensions
        # Player sum: 12-21 (10 states)
        # Dealer showing: 1-10 (10 states)
        # Usable ace: 0-1 (2 states)
        self.player_sums = np.arange(12, 22)
        self.dealer_cards = np.arange(1, 11)
        self.usable_ace = [False, True]
        
    def visualize_state_space_3d(self):
        """Visualize the blackjack state space in 3D"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        X, Y = np.meshgrid(self.dealer_cards, self.player_sums)
        
        # Create sample value function (would be replaced with actual)
        Z = np.zeros_like(X)
        for i, player_sum in enumerate(self.player_sums):
            for j, dealer_card in enumerate(self.dealer_cards):
                # Sample value: higher for better player sums, lower for high dealer cards
                Z[i, j] = (player_sum - 12) * 2 - dealer_card * 0.5
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.8, 
                               linewidth=0.5, antialiased=True, edgecolor='black')
        
        ax.set_xlabel('Dealer Showing Card', fontsize=12)
        ax.set_ylabel('Player Sum', fontsize=12)
        ax.set_zlabel('State Value', fontsize=12)
        ax.set_title('3D Blackjack State Space (No Usable Ace)', fontsize=14, fontweight='bold')
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'blackjack_3d_state_space.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_dual_ace_surfaces(self):
        """Visualize value functions for both usable ace scenarios"""
        fig = plt.figure(figsize=(18, 12))
        
        for ace_idx, has_ace in enumerate([False, True]):
            ax = fig.add_subplot(1, 2, ace_idx + 1, projection='3d')
            
            X, Y = np.meshgrid(self.dealer_cards, self.player_sums)
            Z = np.zeros_like(X)
            
            for i, player_sum in enumerate(self.player_sums):
                for j, dealer_card in enumerate(self.dealer_cards):
                    # Adjust for usable ace
                    effective_sum = player_sum + (10 if has_ace else 0)
                    Z[i, j] = (effective_sum - 12) * 2 - dealer_card * 0.5
            
            surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.8, 
                                   linewidth=0.5, antialiased=True, edgecolor='black')
            
            ax.set_xlabel('Dealer Showing Card', fontsize=12)
            ax.set_ylabel('Player Sum', fontsize=12)
            ax.set_zlabel('State Value', fontsize=12)
            ax.set_title(f'3D Blackjack State Space (Usable Ace: {has_ace})', 
                        fontsize=14, fontweight='bold')
            ax.view_init(elev=30, azim=45)
            
            plt.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, 'blackjack_3d_dual_ace.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_policy_3d(self, policy):
        """Visualize policy as 3D surface"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(self.dealer_cards, self.player_sums)
        Z = np.zeros_like(X)
        
        # Convert policy to surface (0 = hit, 1 = stand)
        for i, player_sum in enumerate(self.player_sums):
            for j, dealer_card in enumerate(self.dealer_cards):
                # Sample policy (would use actual policy)
                Z[i, j] = 1 if player_sum >= 20 else 0
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.8, 
                               linewidth=0.5, antialiased=True, edgecolor='black')
        
        ax.set_xlabel('Dealer Showing Card', fontsize=12)
        ax.set_ylabel('Player Sum', fontsize=12)
        ax.set_zlabel('Action (0=Hit, 1=Stand)', fontsize=12)
        ax.set_title('3D Blackjack Policy', fontsize=14, fontweight='bold')
        ax.view_init(elev=30, azim=45)
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        plt.savefig(os.path.join(IMAGE_DIR, 'blackjack_3d_policy.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main function to demonstrate 3D visualizations"""
    blackjack = Blackjack3D()
    
    print("Visualizing state space...")
    blackjack.visualize_state_space_3d()
    
    print("\nVisualizing dual ace surfaces...")
    blackjack.visualize_dual_ace_surfaces()
    
    print("\nVisualizing policy...")
    # Create sample policy
    policy = np.zeros((10, 10))
    blackjack.visualize_policy_3d(policy)


if __name__ == '__main__':
    main()
