# Interactive 3D Games

This directory contains interactive 3D games based on reinforcement learning environments. Play against AI, control agents, and explore the environments in real-time!

## Available Games

### 1. Tic-Tac-Toe 3D
**File**: `chapter01/tic_tac_toe_3d_game.py`

**Description**: Play Tic-Tac-Toe against an AI opponent in 3D. Click on the board to make your moves!

**Controls**:
- Click on any empty cell to place your X
- AI will automatically respond with O
- Game ends when someone wins or it's a tie

**Modes**:
- Human vs AI: `python chapter01/tic_tac_toe_3d_game.py --mode human`
- AI vs AI: `python chapter01/tic_tac_toe_3d_game.py --mode ai`

**Usage**:
```bash
python chapter01/tic_tac_toe_3d_game.py
```

### 2. Cliff Walking 3D
**File**: `chapter06/cliff_walking_3d_game.py`

**Description**: Navigate an agent from start to goal while avoiding the cliff. The cliff is a dangerous area that resets you to the start!

**Controls**:
- **Arrow Keys**: Move the agent (Up/Down/Left/Right)
- **R**: Reset the game
- **Auto Play Button**: Enable random auto-play

**Goal**: Reach the goal position (blue star) without falling off the cliff (red area)

**Usage**:
```bash
python chapter06/cliff_walking_3d_game.py
```

### 3. Mountain Car 3D
**File**: `chapter10/mountain_car_3d_interactive.py`

**Description**: Control a car stuck in a valley. You need to build momentum by swinging back and forth to reach the top of the mountain!

**Controls**:
- **Left Arrow**: Push left (reverse)
- **Right Arrow**: Push right (forward)
- **Release Keys**: Coast (neutral)
- **R**: Reset the game
- **Auto Play Button**: Enable random auto-play

**Goal**: Reach the top of the mountain (position >= 0.5)

**Physics**: 
- The car has limited power and must use gravity and momentum
- Swing back and forth to build up speed
- This is a challenging control problem!

**Usage**:
```bash
python chapter10/mountain_car_3d_interactive.py
```

### 4. Maze 3D
**File**: `chapter08/maze_3d_game.py`

**Description**: Navigate through a 3D maze with obstacles. Find the optimal path from start to goal!

**Controls**:
- **Arrow Keys**: Move the agent (Up/Down/Left/Right)
- **R**: Reset the game
- **Auto Play Button**: Enable random auto-play

**Goal**: Reach the goal position (blue star) by navigating around obstacles (red bars)

**Usage**:
```bash
python chapter08/maze_3d_game.py
```

## Master Game Launcher

Run all games from a single menu:

```bash
python play_games.py
```

This provides an interactive menu to:
- Select which game to play
- Get instructions for each game
- Exit gracefully

## Game Features

### Interactive Controls
- **Real-time visualization**: See your actions immediately in 3D
- **Keyboard input**: Intuitive arrow key controls
- **Mouse interaction**: Click-based controls for Tic-Tac-Toe
- **Reset functionality**: Start over anytime

### Visual Feedback
- **Path tracking**: See your trajectory in 3D
- **Status displays**: Real-time stats (steps, rewards, position)
- **Color coding**: 
  - Green = Start
  - Blue = Goal
  - Red = Obstacles/Danger
  - Yellow = Current position
  - Colored paths = Trajectory

### Auto-Play Mode
- Watch AI agents play automatically
- Useful for understanding optimal strategies
- Can be toggled on/off during gameplay

## Tips for Playing

### Tic-Tac-Toe
- The AI is trained using temporal difference learning
- Try to block the AI's winning moves
- Look for opportunities to create two threats

### Cliff Walking
- The safe path is along the top row
- Avoid the cliff (row 2, columns 1-10) at all costs!
- Shortest path: Go right along the top, then down at the end

### Mountain Car
- This is intentionally difficult!
- You need to swing back and forth to build momentum
- Don't give up - it takes practice
- Watch the velocity indicator to understand momentum

### Maze
- Plan your route before moving
- Obstacles block your path
- Try to find the shortest route
- Use auto-play to see how AI navigates

## Technical Details

### Requirements
- `matplotlib` with 3D support
- `numpy`
- Original chapter modules (for environment logic)

### Performance
- Games run in real-time with smooth 3D rendering
- Some games use animation for continuous updates
- Auto-play mode may be slower for demonstration

### Integration
- Games use the same environment logic as training scripts
- Can be extended to use trained policies
- Compatible with existing visualization code

## Troubleshooting

### Games won't start
- Make sure matplotlib backend supports interactivity
- Try: `export MPLBACKEND=TkAgg` (Linux/Mac) or use Qt5Agg
- Check that all dependencies are installed

### Controls not responding
- Make sure the game window has focus
- Click on the plot area before using keyboard
- Some systems may require different key codes

### Performance issues
- Reduce animation speed in code
- Close other applications
- Use auto-play mode for smoother experience

## Future Enhancements

Potential additions:
- Multiplayer modes
- Difficulty levels
- Score tracking and leaderboards
- Save/replay functionality
- Integration with trained RL agents
- More games from other chapters

## Notes

- Games are designed to be educational and fun
- They demonstrate RL concepts in interactive ways
- Some games (like Mountain Car) are intentionally challenging
- Auto-play mode helps understand optimal strategies
- All games can be reset and replayed indefinitely

Enjoy playing and learning about reinforcement learning!
