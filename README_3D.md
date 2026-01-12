# 3D Visualizations for Reinforcement Learning

This directory contains interactive 3D visualizations and games for various reinforcement learning environments from Sutton & Barto's book.

## Overview

Each chapter has been enhanced with 3D visualizations that provide:
- **Interactive 3D environments** - See the state spaces in three dimensions
- **Value function surfaces** - Visualize learned values as 3D surfaces
- **Trajectory visualization** - Watch agents explore environments in 3D
- **Policy visualization** - See optimal policies overlaid on value surfaces

## Available 3D Visualizations

### Chapter 1: Tic-Tac-Toe 3D
**File**: `chapter01/tic_tac_toe_3d.py`

- 3D board visualization with X and O pieces
- Value function visualization in 3D space
- Interactive game state display

**Usage**:
```bash
python chapter01/tic_tac_toe_3d.py
```

### Chapter 2: Multi-Armed Bandits 3D
**File**: `chapter02/bandits_3d.py`

- 3D reward distribution visualization
- Learning progress over time as 3D surface
- Action selection landscape

**Usage**:
```bash
python chapter02/bandits_3d.py
```

### Chapter 3: Gridworld 3D
**File**: `chapter03/gridworld_3d.py`

- Value function as 3D surface
- Optimal policy visualization with arrows
- Convergence visualization showing value updates

**Usage**:
```bash
python chapter03/gridworld_3d.py
```

### Chapter 5: Blackjack 3D
**File**: `chapter05/blackjack_3d.py`

- 3D state space visualization
- Dual surfaces for usable ace scenarios
- Policy visualization

**Usage**:
```bash
python chapter05/blackjack_3d.py
```

### Chapter 6: Cliff Walking 3D
**File**: `chapter06/cliff_walking_3d.py`

- 3D environment with cliff visualization
- Agent path tracking in 3D
- Q-value surfaces for each action

**Usage**:
```bash
python chapter06/cliff_walking_3d.py
```

### Chapter 8: Maze 3D
**File**: `chapter08/maze_3d.py`

- 3D maze structure with obstacles
- Agent exploration paths
- Q-value visualization for each action

**Usage**:
```bash
python chapter08/maze_3d.py
```

### Chapter 10: Mountain Car 3D Game
**File**: `chapter10/mountain_car_3d_game.py`

- 3D mountain landscape
- Trajectory visualization
- Value function surface

**Usage**:
```bash
python chapter10/mountain_car_3d_game.py
```

## Master Script

Run all visualizations from a single menu:

```bash
python visualize_all_3d.py
```

This provides an interactive menu to:
- Select individual chapter visualizations
- Run all visualizations sequentially
- Exit gracefully

## Requirements

All 3D visualizations require:
- `matplotlib` (with 3D support)
- `numpy`
- Original chapter modules (for environment logic)

## Features

### Interactive Viewing
- Rotate 3D plots by clicking and dragging
- Zoom in/out with mouse wheel
- Adjust elevation and azimuth angles programmatically

### Export Options
- All visualizations save high-resolution PNG images to `images/` directory
- Images are named with descriptive filenames (e.g., `gridworld_3d_value_function.png`)

### Color Coding
- **Green**: Start positions
- **Red/Orange**: Goal positions or obstacles
- **Blue**: Agent paths or trajectories
- **Color maps**: Value functions use viridis/coolwarm for gradient visualization

## Tips

1. **Performance**: Some visualizations may take a moment to render, especially with large state spaces
2. **Viewing Angles**: Each visualization uses optimized viewing angles, but you can adjust them in the code
3. **Integration**: These visualizations work alongside the original 2D plotting code - they don't replace it
4. **Customization**: Feel free to modify colors, viewing angles, or add additional visualizations

## Future Enhancements

Potential additions:
- Real-time animation of learning progress
- Interactive gameplay (click to take actions)
- Comparison views (multiple algorithms side-by-side)
- Export to interactive HTML (using plotly)

## Notes

- Some visualizations use sample data when actual trained models aren't available
- To use with trained models, modify the visualization scripts to load saved policies/Q-values
- All visualizations are designed to be self-contained and runnable independently
