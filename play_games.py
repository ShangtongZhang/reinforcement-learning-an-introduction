"""
Master script to play interactive 3D games
Provides a menu to select which game to play
"""
import os
import sys

def print_menu():
    """Print the menu of available games"""
    print("\n" + "="*60)
    print("INTERACTIVE 3D REINFORCEMENT LEARNING GAMES")
    print("="*60)
    print("\nAvailable Games:")
    print("  1. Tic-Tac-Toe 3D (Play against AI)")
    print("  2. Cliff Walking 3D (Navigate the cliff)")
    print("  3. Mountain Car 3D (Drive to the top)")
    print("  4. Maze 3D (Navigate the maze)")
    print("  0. Exit")
    print("\n" + "="*60)

def run_tic_tac_toe():
    """Run Tic-Tac-Toe game"""
    print("\nStarting Tic-Tac-Toe 3D Game...")
    try:
        from chapter01.tic_tac_toe_3d_game import TicTacToe3DGame
        game = TicTacToe3DGame(player_vs_ai=True)
        game.run()
    except Exception as e:
        print(f"Error running Tic-Tac-Toe: {e}")
        import traceback
        traceback.print_exc()

def run_cliff_walking():
    """Run Cliff Walking game"""
    print("\nStarting Cliff Walking 3D Game...")
    try:
        from chapter06.cliff_walking_3d_game import CliffWalking3DGame
        game = CliffWalking3DGame()
        game.run()
    except Exception as e:
        print(f"Error running Cliff Walking: {e}")
        import traceback
        traceback.print_exc()

def run_mountain_car():
    """Run Mountain Car game"""
    print("\nStarting Mountain Car 3D Game...")
    try:
        from chapter10.mountain_car_3d_interactive import MountainCar3DInteractive
        game = MountainCar3DInteractive()
        game.run()
    except Exception as e:
        print(f"Error running Mountain Car: {e}")
        import traceback
        traceback.print_exc()

def run_maze():
    """Run Maze game"""
    print("\nStarting Maze 3D Game...")
    try:
        from chapter08.maze_3d_game import Maze3DGame
        game = Maze3DGame()
        game.run()
    except Exception as e:
        print(f"Error running Maze: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main menu loop"""
    while True:
        print_menu()
        try:
            choice = input("\nEnter your choice (0-4): ").strip()
            
            if choice == '0':
                print("\nThanks for playing!")
                break
            elif choice == '1':
                run_tic_tac_toe()
            elif choice == '2':
                run_cliff_walking()
            elif choice == '3':
                run_mountain_car()
            elif choice == '4':
                run_maze()
            else:
                print("\nInvalid choice. Please enter a number between 0-4.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
