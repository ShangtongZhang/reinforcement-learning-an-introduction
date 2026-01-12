"""
Master script to run all 3D visualizations
Provides a menu to select which chapter's 3D visualization to run
"""
import os
import sys

def print_menu():
    """Print the menu of available 3D visualizations"""
    print("\n" + "="*60)
    print("3D VISUALIZATIONS FOR REINFORCEMENT LEARNING")
    print("="*60)
    print("\nAvailable Chapters:")
    print("  1. Chapter 1: Tic-Tac-Toe 3D Game")
    print("  2. Chapter 2: Multi-Armed Bandits 3D")
    print("  3. Chapter 3: Gridworld 3D Value Functions")
    print("  4. Chapter 5: Blackjack 3D State Space")
    print("  5. Chapter 6: Cliff Walking 3D")
    print("  6. Chapter 8: Maze 3D Exploration")
    print("  7. Chapter 10: Mountain Car 3D Game")
    print("  8. Run All Visualizations")
    print("  0. Exit")
    print("\n" + "="*60)

def run_chapter1():
    """Run Chapter 1 3D visualization"""
    print("\nRunning Chapter 1: Tic-Tac-Toe 3D...")
    try:
        from chapter01.tic_tac_toe_3d import main
        main()
    except Exception as e:
        print(f"Error running Chapter 1: {e}")

def run_chapter2():
    """Run Chapter 2 3D visualization"""
    print("\nRunning Chapter 2: Multi-Armed Bandits 3D...")
    try:
        from chapter02.bandits_3d import main
        main()
    except Exception as e:
        print(f"Error running Chapter 2: {e}")

def run_chapter3():
    """Run Chapter 3 3D visualization"""
    print("\nRunning Chapter 3: Gridworld 3D...")
    try:
        from chapter03.gridworld_3d import main
        main()
    except Exception as e:
        print(f"Error running Chapter 3: {e}")

def run_chapter5():
    """Run Chapter 5 3D visualization"""
    print("\nRunning Chapter 5: Blackjack 3D...")
    try:
        from chapter05.blackjack_3d import main
        main()
    except Exception as e:
        print(f"Error running Chapter 5: {e}")

def run_chapter6():
    """Run Chapter 6 3D visualization"""
    print("\nRunning Chapter 6: Cliff Walking 3D...")
    try:
        from chapter06.cliff_walking_3d import main
        main()
    except Exception as e:
        print(f"Error running Chapter 6: {e}")

def run_chapter8():
    """Run Chapter 8 3D visualization"""
    print("\nRunning Chapter 8: Maze 3D...")
    try:
        from chapter08.maze_3d import main
        main()
    except Exception as e:
        print(f"Error running Chapter 8: {e}")

def run_chapter10():
    """Run Chapter 10 3D visualization"""
    print("\nRunning Chapter 10: Mountain Car 3D...")
    try:
        from chapter10.mountain_car_3d_game import main
        main()
    except Exception as e:
        print(f"Error running Chapter 10: {e}")

def run_all():
    """Run all 3D visualizations"""
    print("\nRunning all 3D visualizations...")
    chapters = [
        ("Chapter 1", run_chapter1),
        ("Chapter 2", run_chapter2),
        ("Chapter 3", run_chapter3),
        ("Chapter 5", run_chapter5),
        ("Chapter 6", run_chapter6),
        ("Chapter 8", run_chapter8),
        ("Chapter 10", run_chapter10),
    ]
    
    for name, func in chapters:
        print(f"\n{'='*60}")
        print(f"Running {name}...")
        print('='*60)
        try:
            func()
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    print("\n" + "="*60)
    print("All visualizations completed!")
    print("="*60)

def main():
    """Main menu loop"""
    while True:
        print_menu()
        try:
            choice = input("\nEnter your choice (0-8): ").strip()
            
            if choice == '0':
                print("\nExiting...")
                break
            elif choice == '1':
                run_chapter1()
            elif choice == '2':
                run_chapter2()
            elif choice == '3':
                run_chapter3()
            elif choice == '4':
                run_chapter5()
            elif choice == '5':
                run_chapter6()
            elif choice == '6':
                run_chapter8()
            elif choice == '7':
                run_chapter10()
            elif choice == '8':
                run_all()
            else:
                print("\nInvalid choice. Please enter a number between 0-8.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == '__main__':
    main()
