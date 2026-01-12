# Reinforcement Learning: An Introduction - Enhanced Edition

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Interactive implementation of Sutton & Barto's "Reinforcement Learning: An Introduction (2nd Edition)"**

🎮 **NEW:** Interactive web app, 3D visualizations, and playable games!

![RL Studio](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## 🌟 What's New in This Enhanced Edition

### 🌐 Interactive Web Application
- **4 RL Environments**: Tic-Tac-Toe, Grid World, Cliff Walking, Mountain Car
- **Live Q-Learning**: Train agents in real-time and watch them learn
- **Policy & Value Visualizations**: Heat maps showing learned knowledge
- **Export/Import**: Save and load trained agents
- **Tutorial System**: Step-by-step learning guide

### 🎮 Interactive 3D Games
- **Play against AI** in Tic-Tac-Toe
- **Navigate environments** with arrow keys
- **Real-time physics** in Mountain Car
- **Beautiful 3D graphics** using matplotlib

### 📊 3D Visualizations
- **13 chapters** of enhanced visualizations
- **Value function surfaces** in 3D
- **Policy visualizations** with arrows
- **Learning convergence** animations
- **Trajectory tracking** in 3D space

---

## 🚀 Quick Start

### Option 1: Web App (Recommended)

```bash
# Navigate to web directory
cd web

# Install dependencies
npm install

# Start development server
npm run dev
```

Visit **http://localhost:3000** to play with interactive RL environments!

**No Installation Option:** Open `web/standalone.html` directly in your browser.

### Option 2: Python Scripts

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run any chapter
python chapter06/cliff_walking.py

# Play interactive games
python play_games.py

# View 3D visualizations
python visualize_all_3d.py
```

---

## 📚 Contents

### Original Chapter Implementations

All chapters from Sutton & Barto's textbook, **enhanced with**:
- ✅ Python 3.13 compatibility
- ✅ Robust path handling (run from anywhere)
- ✅ Fixed NumPy 2.x compatibility
- ✅ Performance optimizations

| Chapter | Topics | Files |
|---------|--------|-------|
| **1** | Tic-Tac-Toe | `tic_tac_toe.py`, `tic_tac_toe_3d.py`, `tic_tac_toe_3d_game.py` |
| **2** | Multi-Armed Bandits | `ten_armed_testbed.py`, `bandits_3d.py` |
| **3** | Finite MDPs | `grid_world.py`, `gridworld_3d.py` |
| **4** | Dynamic Programming | `car_rental.py`, `gamblers_problem.py`, `gridworld_car_rental_3d.py` |
| **5** | Monte Carlo Methods | `blackjack.py`, `blackjack_3d.py` |
| **6** | Temporal Difference | `cliff_walking.py`, `cliff_walking_3d.py`, `cliff_walking_3d_game.py` |
| **7** | n-step Bootstrapping | `random_walk.py`, `random_walk_3d.py` |
| **8** | Planning & Learning | `maze.py`, `maze_3d.py`, `maze_3d_game.py` |
| **9** | On-policy Prediction | `random_walk.py`, `function_approximation_3d.py` |
| **10** | On-policy Control | `mountain_car.py`, `mountain_car_3d_game.py` |
| **11** | Off-policy Methods | `counterexample.py`, `counterexample_3d_viz.py` |
| **12** | Eligibility Traces | `random_walk.py`, `mountain_car.py`, `td_lambda_3d.py` |
| **13** | Policy Gradient | `short_corridor.py`, `policy_gradient_3d.py` |

---

## 🎯 Features

### Web Application

#### Environments
- **Tic-Tac-Toe**: Play against Q-Learning AI
- **Grid World**: Navigate grid with obstacles
- **Cliff Walking**: Avoid dangerous cliffs
- **Mountain Car**: Physics-based control problem

#### Learning Features
- **Real-time Training**: Train for 1,000 episodes instantly
- **Policy Visualization**: Heat maps with directional arrows
- **Value Visualization**: Color-coded state values
- **Q-Value Display**: See exact learned values
- **Performance Metrics**: Win/loss rates, steps, rewards

#### Advanced Features
- **Save/Load Agents**: Export/import trained policies
- **Tutorial Mode**: Interactive learning guide
- **Auto-Play**: Watch agents play autonomously
- **Parameter Control**: Adjust α (learning rate) and ε (exploration)
- **Modern UI**: Beautiful, responsive design

### Python Enhancements

#### 3D Visualizations
- Value functions as 3D surfaces
- Policy visualization with arrows
- Learning convergence over time
- Trajectory tracking in 3D
- Multi-algorithm comparisons

#### Interactive Games
- Arrow key controls
- Real-time visualization
- Auto-play mode
- Physics simulation (Mountain Car)
- Click-based gameplay (Tic-Tac-Toe)

---

## 📦 Dependencies

### Python
```
python 3.6+ (tested with 3.13)
numpy
matplotlib
seaborn
tqdm
scipy
```

### Web Application
```
react ^18.2.0
react-dom ^18.2.0
lucide-react ^0.294.0
vite ^5.0.0
tailwindcss ^3.3.6
```

---

## 🎓 Educational Use

### For Students
1. **Start with web app** - Interactive, visual learning
2. **Explore 3D visualizations** - See concepts in 3D
3. **Modify Python scripts** - Experiment with algorithms
4. **Read the code** - Well-documented implementations

### For Instructors
- **Interactive demonstrations** in lectures
- **Homework assignments** using the web app
- **Visual aids** with 3D plots
- **Customizable** for specific topics

### For Researchers
- **Baseline implementations** of classic algorithms
- **Visualization tools** for papers/presentations
- **Extensible codebase** for new algorithms
- **Export functionality** for reproducibility

---

## 🏗️ Project Structure

```
reinforcement-learning-an-introduction/
├── chapter01/ - chapter13/    # Original + enhanced scripts
│   ├── *.py                    # Original implementations
│   ├── *_3d.py                 # 3D visualizations
│   └── *_3d_game.py           # Interactive games
│
├── web/                        # React web application
│   ├── src/
│   │   ├── App.jsx            # Main app
│   │   └── environments/      # Environment components
│   ├── standalone.html        # No-install version
│   └── README.md              # Web app docs
│
├── images/                     # Generated figures
├── venv/                       # Python environment
├── visualize_all_3d.py        # 3D visualization menu
├── play_games.py              # Interactive games menu
├── PROJECT_SUMMARY.md         # Complete project overview
└── README.md                  # This file
```

---

## 📈 Performance Notes

### Quick Executions (< 1 minute)
- Chapters 1, 2, 3: Grid worlds and basic games
- Web app: All training and gameplay

### Moderate (1-5 minutes)
- Chapters 5, 6, 7, 10-13: Monte Carlo, TD methods
- 3D visualizations: Most chapters

### Long (5+ minutes)
- Chapter 4: Car rental (Poisson calculations)
- Chapter 8: Large maze experiments
- Chapter 12: Lambda effect (full experiment ~2 hours)

**Tip**: Most scripts support parameter reduction for faster testing.

---

## 🎨 Screenshots

### Web Application
- **Tic-Tac-Toe**: Play against trained Q-Learning agent
- **Grid World**: Navigate with policy and value heat maps
- **Cliff Walking**: Learn safe vs optimal paths
- **Mountain Car**: Physics-based challenge

### 3D Visualizations
- Value functions as surfaces
- Policy arrows in 3D
- Learning convergence animations
- Multi-algorithm comparisons

*(Screenshots in `images/` directory)*

---

## 🛠️ Development

### Running Tests
```bash
# Test all original scripts
source venv/bin/activate
python chapter02/ten_armed_testbed.py
python chapter06/cliff_walking.py

# Test 3D visualizations
python chapter03/gridworld_3d.py

# Test games
python chapter01/tic_tac_toe_3d_game.py
```

### Web Development
```bash
cd web
npm run dev      # Development server
npm run build    # Production build
npm run preview  # Preview production build
```

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional environments (Atari, MuJoCo)
- Deep RL algorithms (DQN, PPO, A3C)
- More visualizations
- Mobile app version
- Tutorial content

---

## 📄 License

This project builds on the original work by Shangtong Zhang and contributors.
See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Original Code**: [Shangtong Zhang](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
- **Textbook**: Sutton & Barto - "Reinforcement Learning: An Introduction"
- **Libraries**: NumPy, Matplotlib, React, Vite, TailwindCSS, Lucide React

---

## 📞 Support

- **Issues**: Open a GitHub issue
- **Documentation**: See `PROJECT_SUMMARY.md` for complete details
- **Web App Help**: See `web/README.md`
- **Games Guide**: See `README_GAMES.md`
- **3D Visualizations**: See `README_3D.md`

---

## 🎯 Getting Started - Step by Step

### For Complete Beginners

1. **Try the web app** (easiest):
   ```bash
   open web/standalone.html
   ```

2. **Play the tutorial** to learn Q-learning basics

3. **Train an agent** by clicking "Train 1,000 Episodes"

4. **Watch it learn** with value and policy heat maps

### For Python Users

1. **Install dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run a simple example**:
   ```bash
   python chapter03/grid_world.py
   ```

3. **Try 3D visualizations**:
   ```bash
   python visualize_all_3d.py
   ```

4. **Play interactive games**:
   ```bash
   python play_games.py
   ```

---

## 🌟 Highlights

- ✅ **18 3D Python visualizations** created
- ✅ **4 interactive games** with real-time rendering
- ✅ **Full web application** with 4 RL environments
- ✅ **Policy and value heat maps** for visual learning
- ✅ **Tutorial system** for beginners
- ✅ **Export/import** for reproducibility
- ✅ **All Python 3.13 compatible**
- ✅ **Production-ready** code quality

---

**Star ⭐ this repo if you find it useful for learning RL!**

**Live Demo**: `npm run dev` in `web/` directory

**Questions?** Open an issue or check `PROJECT_SUMMARY.md`

---

*Last updated: January 2026 • Python 3.13 • React 18*
