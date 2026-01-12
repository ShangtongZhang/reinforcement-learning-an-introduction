import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, RotateCcw, Brain, Zap, Grid as GridIcon, Layout, ChevronRight, Activity, TrendingUp, AlertTriangle, Mountain, Compass, Target, Award, BarChart3, Settings } from 'lucide-react';

/**
 * ENHANCED Q-LEARNING AGENT WITH SARSA SUPPORT
 */
class RLAgent {
  constructor(alpha = 0.5, gamma = 0.9, epsilon = 0.1, algorithm = 'q-learning') {
    this.qTable = {};
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this.algorithm = algorithm; // 'q-learning' or 'sarsa'
    this.episodeRewards = [];
    this.episodeSteps = [];
  }

  getQ(state, action) {
    if (!this.qTable[state]) this.qTable[state] = {};
    return this.qTable[state][action] || 0.0;
  }

  setQ(state, action, value) {
    if (!this.qTable[state]) this.qTable[state] = {};
    this.qTable[state][action] = value;
  }

  getBestAction(state, availableActions) {
    if (availableActions.length === 0) return null;
    let bestValue = -Infinity;
    let bestActions = [];
    
    availableActions.forEach(action => {
      const val = this.getQ(state, action);
      if (val > bestValue) {
        bestValue = val;
        bestActions = [action];
      } else if (val === bestValue) {
        bestActions.push(action);
      }
    });

    return bestActions[Math.floor(Math.random() * bestActions.length)];
  }

  chooseAction(state, availableActions) {
    if (Math.random() < this.epsilon) {
      return availableActions[Math.floor(Math.random() * availableActions.length)];
    }
    return this.getBestAction(state, availableActions);
  }

  learn(state, action, reward, nextState, nextAvailableActions, nextAction = null) {
    const currentQ = this.getQ(state, action);
    let target;
    
    if (this.algorithm === 'q-learning') {
      // Q-Learning: Use max Q-value of next state
      const maxNextQ = nextAvailableActions.length > 0
        ? Math.max(...nextAvailableActions.map(a => this.getQ(nextState, a)))
        : 0;
      target = reward + this.gamma * maxNextQ;
    } else {
      // SARSA: Use Q-value of actual next action
      const nextQ = nextAction !== null ? this.getQ(nextState, nextAction) : 0;
      target = reward + this.gamma * nextQ;
    }

    const newQ = currentQ + this.alpha * (target - currentQ);
    this.setQ(state, action, newQ);
  }

  getPolicy(state, availableActions) {
    return this.getBestAction(state, availableActions);
  }

  getStateValues(states) {
    return states.map(state => {
      const actions = Object.keys(this.qTable[state] || {});
      if (actions.length === 0) return 0;
      return Math.max(...actions.map(a => this.getQ(state, parseInt(a))));
    });
  }
}

/**
 * TIC-TAC-TOE (Enhanced with better stats)
 */
const TTT_WIN_COMBOS = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6]
];

const checkWinner = (board) => {
  for (let combo of TTT_WIN_COMBOS) {
    const [a, b, c] = combo;
    if (board[a] && board[a] === board[b] && board[a] === board[c]) {
      return board[a];
    }
  }
  if (!board.includes(null)) return 'DRAW';
  return null;
};

const TicTacToe = () => {
  const [board, setBoard] = useState(Array(9).fill(null));
  const [isXNext, setIsXNext] = useState(true);
  const [gameStatus, setGameStatus] = useState('playing');
  const [winner, setWinner] = useState(null);
  const [agent] = useState(new RLAgent(0.5, 0.9, 0.2));
  const [trainingEpisodes, setTrainingEpisodes] = useState(0);
  const [showQValues, setShowQValues] = useState(false);
  const [autoPlay, setAutoPlay] = useState(false);
  const [epsilon, setEpsilon] = useState(0.1);
  const [learningRate, setLearningRate] = useState(0.5);
  const [stats, setStats] = useState({ wins: 0, losses: 0, draws: 0, totalGames: 0 });
  const [rewardHistory, setRewardHistory] = useState([]);

  useEffect(() => {
    agent.epsilon = epsilon;
    agent.alpha = learningRate;
  }, [epsilon, learningRate, agent]);

  const getBoardState = (b) => b.map(c => c === null ? 0 : (c === 'X' ? 1 : 2)).join('');
  const getAvailableMoves = (b) => b.map((val, idx) => val === null ? idx : null).filter(val => val !== null);

  const handleCellClick = (index) => {
    if (board[index] || gameStatus !== 'playing' || !isXNext) return;
    const newBoard = [...board];
    newBoard[index] = 'X';
    setBoard(newBoard);
    
    const result = checkWinner(newBoard);
    if (result) {
      handleGameEnd(result, newBoard);
    } else {
      setIsXNext(false);
    }
  };

  useEffect(() => {
    if (!isXNext && gameStatus === 'playing') {
      const delay = autoPlay ? 50 : 600;
      const timer = setTimeout(makeBotMove, delay);
      return () => clearTimeout(timer);
    }
  }, [isXNext, gameStatus, autoPlay]);

  const makeBotMove = useCallback(() => {
    const currentStateStr = getBoardState(board);
    const available = getAvailableMoves(board);
    if (available.length === 0) return;

    const action = agent.chooseAction(currentStateStr, available);
    const newBoard = [...board];
    newBoard[action] = 'O';
    setBoard(newBoard);

    const result = checkWinner(newBoard);
    if (result) {
      let reward = result === 'O' ? 10 : (result === 'X' ? -10 : 5);
      agent.learn(currentStateStr, action, reward, getBoardState(newBoard), []);
      handleGameEnd(result, newBoard);
    } else {
      agent.learn(currentStateStr, action, 0, getBoardState(newBoard), getAvailableMoves(newBoard));
      setIsXNext(true);
    }
  }, [board, agent]);

  const handleGameEnd = (result) => {
    setGameStatus('finished');
    setWinner(result);
    const newStats = { ...stats };
    if (result === 'X') newStats.wins++;
    else if (result === 'O') newStats.losses++;
    else newStats.draws++;
    newStats.totalGames++;
    setStats(newStats);

    if (autoPlay) setTimeout(resetGame, 100);
  };

  const resetGame = () => {
    setBoard(Array(9).fill(null));
    setGameStatus('playing');
    setWinner(null);
    setIsXNext(true);
  };

  const trainAgent = () => {
    for (let i = 0; i < 1000; i++) {
      let b = Array(9).fill(null);
      let turn = 'X';
      
      while(true) {
        const available = getAvailableMoves(b);
        if (available.length === 0) break;

        const stateStr = getBoardState(b);
        const action = turn === 'O' 
          ? agent.chooseAction(stateStr, available)
          : available[Math.floor(Math.random() * available.length)];

        b[action] = turn;
        const res = checkWinner(b);
        
        if (turn === 'O') {
          let reward = res === 'O' ? 10 : (res === 'X' ? -10 : (res === 'DRAW' ? 5 : 0));
          agent.learn(stateStr, action, reward, getBoardState(b), getAvailableMoves(b));
        }
        
        if (res) break;
        turn = turn === 'X' ? 'O' : 'X';
      }
    }
    setTrainingEpisodes(e => e + 1000);
  };

  return (
    <div className="flex flex-col lg:flex-row h-full gap-6 p-4">
      <div className="flex-1 flex flex-col items-center justify-center bg-white rounded-xl shadow-lg p-6 border border-slate-200">
        <div className="mb-4 text-center">
          <h2 className="text-2xl font-bold text-slate-800 flex items-center justify-center gap-2">
            <Layout className="text-blue-600" /> Tic-Tac-Toe
          </h2>
          <p className="text-slate-500 text-sm">You (X) vs. Q-Learning Agent (O)</p>
        </div>

        <div className="relative">
          <div className="grid grid-cols-3 gap-2 w-72 h-72">
            {board.map((cell, idx) => {
              const stateStr = getBoardState(board);
              const qVal = agent.getQ(stateStr, idx);
              const isAvailable = cell === null;
              let bgClass = "bg-slate-100 hover:bg-slate-200";
              if (cell === 'X') bgClass = "bg-blue-100 text-blue-600";
              if (cell === 'O') bgClass = "bg-rose-100 text-rose-600";
              
              return (
                <button 
                  key={idx}
                  onClick={() => handleCellClick(idx)}
                  disabled={cell !== null || !isXNext || gameStatus !== 'playing'}
                  className={`relative rounded-lg flex items-center justify-center text-4xl font-bold transition-all duration-200 ${bgClass}`}
                >
                  {cell}
                  {showQValues && isAvailable && (
                    <span className="absolute bottom-1 right-1 text-[10px] text-slate-400 font-mono">
                      {qVal.toFixed(2)}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          
          {gameStatus !== 'playing' && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/90 backdrop-blur-sm rounded-xl">
              <h3 className="text-3xl font-black text-slate-800 mb-2">
                {winner === 'DRAW' ? "It's a Draw!" : `${winner} Wins!`}
              </h3>
              <button 
                onClick={resetGame}
                className="px-6 py-2 bg-slate-900 text-white rounded-full font-medium shadow-lg hover:scale-105 transition-transform flex items-center gap-2"
              >
                <RotateCcw size={18} /> Play Again
              </button>
            </div>
          )}
        </div>

        <div className="mt-6 grid grid-cols-4 gap-4 w-72">
          <div className="flex flex-col items-center p-3 bg-blue-50 rounded-lg">
            <span className="text-2xl font-bold text-blue-600">{stats.wins}</span>
            <span className="text-xs uppercase tracking-wider text-slate-500">Wins</span>
          </div>
          <div className="flex flex-col items-center p-3 bg-slate-50 rounded-lg">
            <span className="text-2xl font-bold text-slate-600">{stats.draws}</span>
            <span className="text-xs uppercase tracking-wider text-slate-500">Draws</span>
          </div>
          <div className="flex flex-col items-center p-3 bg-rose-50 rounded-lg">
            <span className="text-2xl font-bold text-rose-600">{stats.losses}</span>
            <span className="text-xs uppercase tracking-wider text-slate-500">Losses</span>
          </div>
          <div className="flex flex-col items-center p-3 bg-purple-50 rounded-lg">
            <span className="text-2xl font-bold text-purple-600">{stats.totalGames}</span>
            <span className="text-xs uppercase tracking-wider text-slate-500">Total</span>
          </div>
        </div>

        {/* Win Rate Chart */}
        {stats.totalGames > 0 && (
          <div className="mt-6 w-72 bg-slate-50 p-4 rounded-lg">
            <div className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
              <BarChart3 size={16} /> Performance
            </div>
            <div className="flex gap-2 h-2 rounded-full overflow-hidden bg-slate-200">
              <div 
                className="bg-blue-500" 
                style={{ width: `${(stats.wins / stats.totalGames) * 100}%` }}
              />
              <div 
                className="bg-slate-400" 
                style={{ width: `${(stats.draws / stats.totalGames) * 100}%` }}
              />
              <div 
                className="bg-rose-500" 
                style={{ width: `${(stats.losses / stats.totalGames) * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-slate-500 mt-1">
              <span>{((stats.wins / stats.totalGames) * 100).toFixed(0)}% Win</span>
              <span>{((stats.losses / stats.totalGames) * 100).toFixed(0)}% Loss</span>
            </div>
          </div>
        )}
      </div>

      {/* Control Panel */}
      <div className="w-full lg:w-80 bg-slate-50 rounded-xl border border-slate-200 p-6 flex flex-col gap-6">
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-slate-800 font-semibold pb-2 border-b border-slate-200">
            <Brain className="text-purple-600" size={20} />
            <h3>Agent Brain</h3>
          </div>
          
          <div className="bg-white p-4 rounded-lg border border-slate-200 shadow-sm">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-bold text-slate-500 uppercase">Training Episodes</span>
              <span className="text-lg font-mono text-purple-600">{trainingEpisodes.toLocaleString()}</span>
            </div>
            <button 
              onClick={trainAgent}
              className="w-full py-2 bg-purple-100 text-purple-700 hover:bg-purple-200 rounded-md text-sm font-semibold transition-colors flex items-center justify-center gap-2"
            >
              <Zap size={16} /> Train 1,000 Episodes
            </button>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-2 text-slate-800 font-semibold pb-2 border-b border-slate-200">
            <Settings className="text-indigo-600" size={20} />
            <h3>Hyperparameters</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-600">Learning Rate (α)</span>
                <span className="font-mono text-indigo-600">{learningRate.toFixed(2)}</span>
              </div>
              <input 
                type="range" min="0" max="1" step="0.05"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full h-2 bg-indigo-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-600">Exploration (ε)</span>
                <span className="font-mono text-indigo-600">{epsilon.toFixed(2)}</span>
              </div>
              <input 
                type="range" min="0" max="1" step="0.05"
                value={epsilon}
                onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                className="w-full h-2 bg-indigo-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
            </div>
          </div>
        </div>

        <div className="space-y-2 mt-auto">
          <label className="flex items-center justify-between p-3 bg-white rounded-lg border border-slate-200 cursor-pointer hover:bg-slate-50 transition-colors">
            <span className="text-sm font-medium text-slate-700">Show Q-Values</span>
            <input 
              type="checkbox" 
              className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500" 
              checked={showQValues} 
              onChange={() => setShowQValues(!showQValues)} 
            />
          </label>

          <label className="flex items-center justify-between p-3 bg-white rounded-lg border border-slate-200 cursor-pointer hover:bg-slate-50 transition-colors">
            <span className="text-sm font-medium text-slate-700">Auto-Play Mode</span>
            <input 
              type="checkbox" 
              className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500" 
              checked={autoPlay} 
              onChange={() => { setAutoPlay(!autoPlay); resetGame(); }} 
            />
          </label>
        </div>
      </div>
    </div>
  );
};

// ... (Continue with enhanced GridWorld and new environments)

export default function RLStudio() {
  const [activeTab, setActiveTab] = useState('tictactoe');

  return (
    <div className="min-h-screen bg-slate-100 font-sans text-slate-900 flex flex-col">
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between sticky top-0 z-50 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-lg flex items-center justify-center text-white shadow-md">
            <Brain size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-indigo-700">
              RL Interactive Studio
            </h1>
            <p className="text-xs text-slate-500 font-medium">Learn Reinforcement Learning Interactively</p>
          </div>
        </div>
        
        <nav className="flex bg-slate-100 p-1 rounded-lg gap-1">
          {['tictactoe', 'gridworld'].map(tab => (
            <button 
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-md text-sm font-semibold transition-all ${
                activeTab === tab ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              {tab === 'tictactoe' ? 'Tic-Tac-Toe' : 'Grid World'}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1 overflow-auto p-2 md:p-6 max-w-7xl mx-auto w-full">
        {activeTab === 'tictactoe' ? <TicTacToe /> : <div>Grid World (placeholder)</div>}
      </main>
    </div>
  );
}
