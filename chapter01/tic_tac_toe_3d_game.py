"""
Interactive 3D Tic-Tac-Toe Game
Play against AI or watch AI vs AI in 3D
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import os
import sys
import pickle

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter01.tic_tac_toe import State, Player, Judger, HumanPlayer, all_states, BOARD_ROWS, BOARD_COLS

# Setup image directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'images')

class TicTacToe3DGame:
    def __init__(self, player_vs_ai=True):
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.current_state = State()
        self.player1 = None
        self.player2 = None
        self.judger = None
        self.player_vs_ai = player_vs_ai
        self.game_over = False
        self.current_player = 1
        self.click_pos = None
        self.board_size = 3
        
        # Setup game
        if player_vs_ai:
            self.player1 = HumanPlayer3D()
            self.player2 = Player(epsilon=0)
            try:
                self.player2.load_policy()
            except:
                print("No saved policy found. Training AI first...")
                self.train_ai()
                self.player2 = Player(epsilon=0)
                self.player2.load_policy()
        else:
            self.player1 = Player(epsilon=0.1)
            self.player2 = Player(epsilon=0.1)
            try:
                self.player1.load_policy()
                self.player2.load_policy()
            except:
                print("No saved policies. Training AIs...")
                self.train_ai()
                self.player1 = Player(epsilon=0.1)
                self.player2 = Player(epsilon=0.1)
                self.player1.load_policy()
                self.player2.load_policy()
        
        self.judger = Judger(self.player1, self.player2)
        self.judger.reset()
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def train_ai(self, epochs=10000):
        """Train AI players"""
        print("Training AI players...")
        player1 = Player(epsilon=0.01)
        player2 = Player(epsilon=0.01)
        judger = Judger(player1, player2)
        
        for i in range(epochs):
            winner = judger.play()
            player1.backup()
            player2.backup()
            judger.reset()
            if (i + 1) % 1000 == 0:
                print(f"Training: {i+1}/{epochs}")
        
        player1.save_policy()
        player2.save_policy()
        print("Training complete!")
        
    def draw_board_3d(self):
        """Draw the 3D tic-tac-toe board"""
        self.ax.clear()
        
        spacing = 1.0
        
        # Draw grid lines
        for i in range(self.board_size + 1):
            # Vertical lines
            self.ax.plot([i, i], [0, self.board_size], [0, 0], 'k-', linewidth=3)
            self.ax.plot([i, i], [0, self.board_size], [0.1, 0.1], 'k-', linewidth=3)
            # Horizontal lines
            self.ax.plot([0, self.board_size], [i, i], [0, 0], 'k-', linewidth=3)
            self.ax.plot([0, self.board_size], [i, i], [0.1, 0.1], 'k-', linewidth=3)
            # Vertical connectors
            self.ax.plot([i, i], [0, 0], [0, 0.1], 'k-', linewidth=3)
            self.ax.plot([i, i], [self.board_size, self.board_size], [0, 0.1], 'k-', linewidth=3)
        
        # Draw pieces
        for i in range(self.board_size):
            for j in range(self.board_size):
                x, y = j + 0.5, i + 0.5
                z = 0.05
                
                if self.current_state.data[i, j] == 1:  # Player 1 (X)
                    size = 0.3
                    self.ax.plot([x-size, x+size], [y-size, y+size], [z, z], 'r-', linewidth=5)
                    self.ax.plot([x-size, x+size], [y+size, y-size], [z, z], 'r-', linewidth=5)
                elif self.current_state.data[i, j] == -1:  # Player 2 (O)
                    theta = np.linspace(0, 2*np.pi, 50)
                    radius = 0.3
                    circle_x = x + radius * np.cos(theta)
                    circle_y = y + radius * np.sin(theta)
                    circle_z = np.full_like(theta, z)
                    self.ax.plot(circle_x, circle_y, circle_z, 'b-', linewidth=5)
        
        # Add status text
        status = self.get_status_text()
        self.ax.text2D(0.05, 0.95, status, transform=self.ax.transAxes, 
                       fontsize=14, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Set labels and title
        self.ax.set_xlabel('Column', fontsize=12)
        self.ax.set_ylabel('Row', fontsize=12)
        self.ax.set_zlabel('Height', fontsize=12)
        title = '3D Tic-Tac-Toe - Click on board to play!' if self.player_vs_ai else '3D Tic-Tac-Toe - AI vs AI'
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set limits
        self.ax.set_xlim([-0.5, self.board_size + 0.5])
        self.ax.set_ylim([-0.5, self.board_size + 0.5])
        self.ax.set_zlim([-0.1, 0.3])
        
        # Set viewing angle
        self.ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        
    def get_status_text(self):
        """Get status text for display"""
        if self.game_over:
            winner = self.current_state.winner
            if winner == 1:
                return "Game Over: Player 1 (X) Wins!"
            elif winner == -1:
                return "Game Over: Player 2 (O) Wins!"
            else:
                return "Game Over: It's a Tie!"
        else:
            if self.current_player == 1:
                return "Player 1 (X) to move"
            else:
                return "Player 2 (O) to move"
    
    def on_click(self, event):
        """Handle mouse clicks on the board"""
        if self.game_over or not self.player_vs_ai:
            return
            
        if event.inaxes != self.ax:
            return
            
        # Convert 2D click to 3D coordinates
        # Get the clicked position in data coordinates
        xdata, ydata = event.xdata, event.ydata
        
        if xdata is None or ydata is None:
            return
        
        # Convert to board coordinates
        col = int(xdata)
        row = int(ydata)
        
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            # Check if position is empty
            if self.current_state.data[row, col] == 0:
                # Make move
                new_state = self.current_state.next_state(row, col, 1)  # Player 1 is X
                self.current_state = new_state
                self.player1.set_state(self.current_state)
                self.player2.set_state(self.current_state)
                
                # Check if game is over
                if self.current_state.is_end():
                    self.game_over = True
                    self.draw_board_3d()
                    plt.draw()
                    return
                
                # AI move
                if not self.current_state.is_end():
                    i, j, symbol = self.player2.act()
                    new_state = self.current_state.next_state(i, j, symbol)
                    self.current_state = new_state
                    self.player1.set_state(self.current_state)
                    self.player2.set_state(self.current_state)
                    
                    if self.current_state.is_end():
                        self.game_over = True
                
                self.draw_board_3d()
                plt.draw()
    
    def play_ai_vs_ai(self):
        """Play AI vs AI automatically"""
        self.current_state = State()
        self.player1.set_state(self.current_state)
        self.player2.set_state(self.current_state)
        self.game_over = False
        
        while not self.game_over:
            # Player 1 move
            i, j, symbol = self.player1.act()
            new_state = self.current_state.next_state(i, j, symbol)
            self.current_state = new_state
            self.player1.set_state(self.current_state)
            self.player2.set_state(self.current_state)
            
            self.draw_board_3d()
            plt.draw()
            plt.pause(0.5)
            
            if self.current_state.is_end():
                self.game_over = True
                break
            
            # Player 2 move
            i, j, symbol = self.player2.act()
            new_state = self.current_state.next_state(i, j, symbol)
            self.current_state = new_state
            self.player1.set_state(self.current_state)
            self.player2.set_state(self.current_state)
            
            self.draw_board_3d()
            plt.draw()
            plt.pause(0.5)
            
            if self.current_state.is_end():
                self.game_over = True
        
        self.draw_board_3d()
        plt.draw()
    
    def reset_game(self):
        """Reset the game"""
        self.current_state = State()
        self.player1.set_state(self.current_state)
        self.player2.set_state(self.current_state)
        self.game_over = False
        self.current_player = 1
        self.draw_board_3d()
        plt.draw()
    
    def run(self):
        """Run the game"""
        self.draw_board_3d()
        
        if self.player_vs_ai:
            print("\nClick on the board to make your move!")
            print("You are Player 1 (X), AI is Player 2 (O)")
            plt.show()
        else:
            print("\nWatching AI vs AI...")
            self.play_ai_vs_ai()
            plt.show()


class HumanPlayer3D:
    """Human player for 3D game (uses click events)"""
    def __init__(self):
        self.symbol = 1
        self.state = None
    
    def reset(self):
        pass
    
    def set_state(self, state):
        self.state = state
    
    def set_symbol(self, symbol):
        self.symbol = symbol
    
    def act(self):
        # This will be handled by click events
        return None


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='3D Tic-Tac-Toe Game')
    parser.add_argument('--mode', choices=['human', 'ai'], default='human',
                       help='Game mode: human vs AI or AI vs AI')
    args = parser.parse_args()
    
    game = TicTacToe3DGame(player_vs_ai=(args.mode == 'human'))
    game.run()


if __name__ == '__main__':
    main()
