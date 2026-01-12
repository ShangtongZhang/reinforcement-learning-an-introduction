# RL Interactive Studio - Web Application

This directory contains an interactive web-based Reinforcement Learning studio for learning RL concepts through play.

## 🚀 Quick Start (Standalone Version)

The easiest way to run the application is using the standalone HTML file:

```bash
open standalone.html
```

This file includes everything needed and runs directly in your browser - no installation required!

## 📦 Full Development Version

If you want the full development setup with Vite and proper tooling:

### Prerequisites
- Node.js 16+ installed
- At least 500MB of free disk space

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## 🎮 Features

### Tic-Tac-Toe with Q-Learning
- **Play against AI**: Test your skills against a learning agent
- **Watch the AI learn**: Train the agent with 1,000 episodes instantly
- **Visualize Q-Values**: See the agent's value estimates for each move
- **Track performance**: Monitor win/loss/draw statistics
- **Adjust parameters**: Control learning rate and exploration rate in real-time

### Key Concepts Demonstrated
- **Q-Learning algorithm**: Value-based reinforcement learning
- **Exploration vs Exploitation**: Epsilon-greedy policy
- **Temporal Difference learning**: Learning from experience
- **Policy improvement**: Agent gets better with training

## 🎨 UI Features

- **Modern, responsive design**: Works on desktop, tablet, and mobile
- **Real-time updates**: See changes immediately
- **Smooth animations**: Polished user experience
- **Interactive controls**: Sliders, toggles, and buttons
- **Performance dashboard**: Visual statistics and progress bars

## 🧠 How It Works

### Q-Learning Algorithm

The agent learns by maintaining a Q-table that stores value estimates for each state-action pair:

```
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
```

Where:
- **α (alpha)**: Learning rate - how quickly the agent updates its knowledge
- **γ (gamma)**: Discount factor - how much the agent values future rewards
- **ε (epsilon)**: Exploration rate - probability of taking a random action

### Training Process

1. **Initialization**: Q-values start at 0
2. **Episode**: Agent plays a complete game
3. **Update**: Q-values are updated based on rewards
4. **Repeat**: Process repeats for many episodes
5. **Convergence**: Agent learns optimal policy

## 🔧 Customization

### Hyperparameters

Adjust these to see how they affect learning:

- **Learning Rate (α)**: 0.01 - 1.0
  - Higher = faster learning but less stable
  - Lower = slower but more stable learning

- **Exploration Rate (ε)**: 0.0 - 1.0
  - Higher = more random moves (exploration)
  - Lower = more greedy moves (exploitation)

### Training

- Click "Train 1,000 Episodes" to rapidly improve the agent
- The agent plays against a random opponent during training
- Training is non-blocking and shows progress

## 📊 Statistics Tracked

- **Wins**: Games you won
- **Losses**: Games the agent won
- **Draws**: Tied games
- **Total**: All games played
- **Win Rate**: Percentage of games won

## 🌐 Browser Compatibility

Works in all modern browsers:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 💡 Tips

1. **Train first**: Let the agent train for a few thousand episodes before playing
2. **Lower epsilon**: Reduce exploration rate to 0.05-0.10 for better play
3. **Watch Q-values**: Enable to see which moves the agent thinks are best
4. **Experiment**: Try different learning rates to see the effect

## 🐛 Troubleshooting

### Standalone version won't open
- Right-click the file and select "Open With" → your browser
- Or drag the file into an open browser window

### npm install fails
- Check disk space (need ~500MB)
- Try: `npm install --legacy-peer-deps`
- Use the standalone version instead

### Agent plays poorly
- Click "Train 1,000 Episodes" multiple times
- Lower the exploration rate
- The agent needs experience to learn

## 📚 Learn More

This demo implements concepts from:
- Sutton & Barto's "Reinforcement Learning: An Introduction"
- Q-Learning (Watkins, 1989)
- Temporal Difference Learning

## 🚀 Future Enhancements

Planned features:
- Grid World environment
- Cliff Walking
- Mountain Car
- Policy visualization
- Training progress charts
- Save/load trained agents
- Multiple algorithm comparisons (SARSA vs Q-Learning)

## 📝 License

Part of the Reinforcement Learning codebase. See main LICENSE file.

## 🤝 Contributing

Feel free to add new environments or improve the UI!

## ❓ Questions?

Open an issue in the main repository for support.
