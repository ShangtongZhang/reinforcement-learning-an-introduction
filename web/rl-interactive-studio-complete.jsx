import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { 
  Play, Pause, RotateCcw, Brain, Zap, Grid as GridIcon, Layout, ChevronRight, 
  Activity, TrendingUp, AlertTriangle, Mountain, Compass, Target, Award, 
  BarChart3, Settings, Info, Download, Upload, Sparkles, X, Check,
  ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Map, Trophy
} from 'lucide-react';

/**
 * ============================================================================
 * CORE RL ENGINE
 * ============================================================================
 */

class RLAgent {
  constructor(alpha = 0.5, gamma = 0.9, epsilon = 0.1, algorithm = 'q-learning') {
    this.qTable = {};
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this.algorithm = algorithm;
    this.visitCounts = {};
    this.episodeRewards = [];
    this.episodeSteps = [];
  }

  getQ(state, action) {
    const key = `${state}_${action}`;
    return this.qTable[key] || 0.0;
  }

  setQ(state, action, value) {
    const key = `${state}_${action}`;
    this.qTable[key] = value;
  }

  incrementVisit(state) {
    this.visitCounts[state] = (this.visitCounts[state] || 0) + 1;
  }

  getVisitCount(state) {
    return this.visitCounts[state] || 0;
  }

  getBestAction(state, availableActions) {
    if (!availableActions || availableActions.length === 0) return null;
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
      const maxNextQ = nextAvailableActions && nextAvailableActions.length > 0
        ? Math.max(...nextAvailableActions.map(a => this.getQ(nextState, a)))
        : 0;
      target = reward + this.gamma * maxNextQ;
    } else {
      const nextQ = nextAction !== null ? this.getQ(nextState, nextAction) : 0;
      target = reward + this.gamma * nextQ;
    }

    const newQ = currentQ + this.alpha * (target - currentQ);
    this.setQ(state, action, newQ);
    this.incrementVisit(state);
  }

  exportPolicy() {
    return {
      qTable: this.qTable,
      visitCounts: this.visitCounts,
      config: {
        alpha: this.alpha,
        gamma: this.gamma,
        epsilon: this.epsilon,
        algorithm: this.algorithm
      }
    };
  }

  importPolicy(data) {
    this.qTable = data.qTable || {};
    this.visitCounts = data.visitCounts || {};
    if (data.config) {
      this.alpha = data.config.alpha;
      this.gamma = data.config.gamma;
      this.epsilon = data.config.epsilon;
      this.algorithm = data.config.algorithm;
    }
  }

  reset() {
    this.qTable = {};
    this.visitCounts = {};
    this.episodeRewards = [];
    this.episodeSteps = [];
  }
}

/**
 * ============================================================================
 * TIC-TAC-TOE ENVIRONMENT
 * ============================================================================
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
  const [agent] = useState(() => new RLAgent(0.5, 0.9, 0.15));
  const [trainingEpisodes, setTrainingEpisodes] = useState(0);
  const [showQValues, setShowQValues] = useState(false);
  const [autoPlay, setAutoPlay] = useState(false);
  const [epsilon, setEpsilon] = useState(0.15);
  const [learningRate, setLearningRate] = useState(0.5);
  const [stats, setStats] = useState({ wins: 0, losses: 0, draws: 0, totalGames: 0 });
  const [trainingProgress, setTrainingProgress] = useState([]);
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    agent.epsilon = epsilon;
    agent.alpha = learningRate;
  }, [epsilon, learningRate]);

  const getBoardState = (b) => b.map(c => c === null ? '0' : (c === 'X' ? '1' : '2')).join('');
  const getAvailableMoves = (b) => b.map((val, idx) => val === null ? idx : null).filter(val => val !== null);

  const handleCellClick = (index) => {
    if (board[index] || gameStatus !== 'playing' || !isXNext || autoPlay) return;
    const newBoard = [...board];
    newBoard[index] = 'X';
    setBoard(newBoard);
    
    const result = checkWinner(newBoard);
    if (result) {
      handleGameEnd(result);
    } else {
      setIsXNext(false);
    }
  };

  useEffect(() => {
    if (!isXNext && gameStatus === 'playing' && !isTraining) {
      const delay = autoPlay ? 50 : 600;
      const timer = setTimeout(makeBotMove, delay);
      return () => clearTimeout(timer);
    }
  }, [isXNext, gameStatus, autoPlay, isTraining]);

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
      handleGameEnd(result);
    } else {
      agent.learn(currentStateStr, action, -0.1, getBoardState(newBoard), getAvailableMoves(newBoard));
      setIsXNext(true);
    }
  }, [board]);

  const handleGameEnd = (result) => {
    setGameStatus('finished');
    setWinner(result);
    const newStats = { ...stats };
    if (result === 'X') newStats.wins++;
    else if (result === 'O') newStats.losses++;
    else newStats.draws++;
    newStats.totalGames++;
    setStats(newStats);

    if (autoPlay) {
      setTimeout(resetGame, 100);
    }
  };

  const resetGame = () => {
    setBoard(Array(9).fill(null));
    setGameStatus('playing');
    setWinner(null);
    setIsXNext(true);
  };

  const trainAgent = async () => {
    setIsTraining(true);
    const batchSize = 100;
    const totalBatches = 10;
    const progress = [];

    for (let batch = 0; batch < totalBatches; batch++) {
      let batchWins = 0, batchLosses = 0, batchDraws = 0;

      for (let i = 0; i < batchSize; i++) {
        let b = Array(9).fill(null);
        let turn = Math.random() > 0.5 ? 'X' : 'O';
        
        while(true) {
          const available = getAvailableMoves(b);
          if (available.length === 0) {
            batchDraws++;
            break;
          }

          const stateStr = getBoardState(b);
          const action = turn === 'O' 
            ? agent.chooseAction(stateStr, available)
            : available[Math.floor(Math.random() * available.length)];

          b[action] = turn;
          const res = checkWinner(b);
          
          if (turn === 'O') {
            let reward = res === 'O' ? 10 : (res === 'X' ? -10 : (res === 'DRAW' ? 5 : -0.1));
            agent.learn(stateStr, action, reward, getBoardState(b), getAvailableMoves(b));
          }
          
          if (res) {
            if (res === 'O') batchWins++;
            else if (res === 'X') batchLosses++;
            else batchDraws++;
            break;
          }
          turn = turn === 'X' ? 'O' : 'X';
        }
      }

      progress.push({
        episode: (batch + 1) * batchSize,
        winRate: batchWins / batchSize,
        lossRate: batchLosses / batchSize,
        drawRate: batchDraws / batchSize
      });
      
      setTrainingProgress([...progress]);
      await new Promise(resolve => setTimeout(resolve, 10));
    }

    setTrainingEpisodes(e => e + 1000);
    setIsTraining(false);
  };

  const winRate = stats.totalGames > 0 ? (stats.wins / stats.totalGames * 100).toFixed(1) : 0;

  return (
    <div className="flex flex-col xl:flex-row h-full gap-6">
      {/* Game Board */}
      <div className="flex-1 flex flex-col items-center justify-center bg-gradient-to-br from-white to-slate-50 rounded-2xl shadow-xl p-8 border border-slate-200">
        <div className="mb-6 text-center">
          <h2 className="text-3xl font-bold text-slate-800 flex items-center justify-center gap-3 mb-2">
            <Layout className="text-blue-600" size={32} /> Tic-Tac-Toe
          </h2>
          <p className="text-slate-500 text-sm font-medium">You (X) vs Q-Learning Agent (O)</p>
        </div>

        <div className="relative">
          <div className="grid grid-cols-3 gap-3 p-4 bg-white rounded-xl shadow-lg border-2 border-slate-200">
            {board.map((cell, idx) => {
              const stateStr = getBoardState(board);
              const qVal = agent.getQ(stateStr, idx);
              const isAvailable = cell === null;
              let bgClass = "bg-gradient-to-br from-slate-50 to-slate-100 hover:from-slate-100 hover:to-slate-200";
              if (cell === 'X') bgClass = "bg-gradient-to-br from-blue-100 to-blue-200 text-blue-700";
              if (cell === 'O') bgClass = "bg-gradient-to-br from-rose-100 to-rose-200 text-rose-700";
              
              return (
                <button 
                  key={idx}
                  onClick={() => handleCellClick(idx)}
                  disabled={cell !== null || !isXNext || gameStatus !== 'playing' || autoPlay}
                  className={`
                    relative w-24 h-24 rounded-xl flex items-center justify-center 
                    text-5xl font-black transition-all duration-200 shadow-md
                    hover:shadow-lg hover:scale-105 disabled:hover:scale-100
                    ${bgClass}
                  `}
                >
                  {cell}
                  {showQValues && isAvailable && qVal !== 0 && (
                    <span className="absolute top-1 right-1 px-1.5 py-0.5 bg-black/10 rounded text-[9px] text-slate-600 font-mono font-bold">
                      {qVal.toFixed(1)}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          
          {gameStatus !== 'playing' && !autoPlay && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/95 backdrop-blur-md rounded-xl">
              <div className={`mb-4 ${winner === 'X' ? 'text-blue-600' : winner === 'O' ? 'text-rose-600' : 'text-slate-600'}`}>
                {winner === 'DRAW' ? <Trophy size={64} /> : <Award size={64} />}
              </div>
              <h3 className="text-4xl font-black text-slate-900 mb-4">
                {winner === 'DRAW' ? "It's a Draw!" : `${winner} Wins!`}
              </h3>
              <button 
                onClick={resetGame}
                className="px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-full font-bold shadow-lg hover:shadow-xl hover:scale-105 transition-all flex items-center gap-2"
              >
                <RotateCcw size={20} /> Play Again
              </button>
            </div>
          )}
        </div>

        {/* Stats Dashboard */}
        <div className="mt-8 grid grid-cols-4 gap-3 w-full max-w-md">
          <div className="flex flex-col items-center p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl shadow-sm">
            <span className="text-3xl font-black text-blue-600">{stats.wins}</span>
            <span className="text-xs uppercase tracking-widest text-blue-700 font-bold">Wins</span>
          </div>
          <div className="flex flex-col items-center p-4 bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl shadow-sm">
            <span className="text-3xl font-black text-slate-600">{stats.draws}</span>
            <span className="text-xs uppercase tracking-widest text-slate-700 font-bold">Draws</span>
          </div>
          <div className="flex flex-col items-center p-4 bg-gradient-to-br from-rose-50 to-rose-100 rounded-xl shadow-sm">
            <span className="text-3xl font-black text-rose-600">{stats.losses}</span>
            <span className="text-xs uppercase tracking-widest text-rose-700 font-bold">Losses</span>
          </div>
          <div className="flex flex-col items-center p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl shadow-sm">
            <span className="text-3xl font-black text-purple-600">{stats.totalGames}</span>
            <span className="text-xs uppercase tracking-widest text-purple-700 font-bold">Total</span>
          </div>
        </div>

        {/* Win Rate Visualization */}
        {stats.totalGames > 0 && (
          <div className="mt-6 w-full max-w-md bg-white p-6 rounded-xl shadow-lg border border-slate-200">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-bold text-slate-700 flex items-center gap-2">
                <BarChart3 size={18} className="text-blue-600" /> Performance
              </span>
              <span className="text-2xl font-black text-blue-600">{winRate}%</span>
            </div>
            <div className="flex gap-1 h-3 rounded-full overflow-hidden bg-slate-100">
              <div 
                className="bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-500" 
                style={{ width: `${(stats.wins / stats.totalGames) * 100}%` }}
              />
              <div 
                className="bg-gradient-to-r from-slate-400 to-slate-500 transition-all duration-500" 
                style={{ width: `${(stats.draws / stats.totalGames) * 100}%` }}
              />
              <div 
                className="bg-gradient-to-r from-rose-500 to-rose-600 transition-all duration-500" 
                style={{ width: `${(stats.losses / stats.totalGames) * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-slate-500 mt-2 font-semibold">
              <span>{((stats.wins / stats.totalGames) * 100).toFixed(0)}% Win</span>
              <span>{((stats.draws / stats.totalGames) * 100).toFixed(0)}% Draw</span>
              <span>{((stats.losses / stats.totalGames) * 100).toFixed(0)}% Loss</span>
            </div>
          </div>
        )}
      </div>

      {/* Control Panel */}
      <div className="w-full xl:w-96 bg-white rounded-2xl border border-slate-200 shadow-xl p-6 flex flex-col gap-6 overflow-y-auto max-h-[900px]">
        {/* Agent Brain Section */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-slate-800 font-bold pb-3 border-b-2 border-purple-100">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Brain className="text-purple-600" size={20} />
            </div>
            <h3 className="text-lg">Agent Brain</h3>
          </div>
          
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-5 rounded-xl border border-purple-200">
            <div className="flex justify-between items-center mb-3">
              <span className="text-xs font-bold text-purple-700 uppercase tracking-wider">Training Episodes</span>
              <span className="text-2xl font-black text-purple-700 font-mono">{trainingEpisodes.toLocaleString()}</span>
            </div>
            <button 
              onClick={trainAgent}
              disabled={isTraining}
              className="w-full py-3 bg-gradient-to-r from-purple-600 to-purple-700 text-white hover:from-purple-700 hover:to-purple-800 rounded-lg text-sm font-bold transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isTraining ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  <Zap size={16} /> Train 1,000 Episodes
                </>
              )}
            </button>
            {isTraining && (
              <div className="mt-3 p-2 bg-white/50 rounded-lg">
                <div className="h-1 bg-purple-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-purple-600 to-purple-700 transition-all duration-300"
                    style={{ width: `${(trainingProgress.length / 10) * 100}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {trainingProgress.length > 0 && (
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
              <h4 className="text-xs font-bold text-slate-600 uppercase mb-3 flex items-center gap-2">
                <TrendingUp size={14} /> Training Progress
              </h4>
              <div className="space-y-2">
                {trainingProgress.slice(-3).map((p, i) => (
                  <div key={i} className="flex justify-between items-center text-xs">
                    <span className="text-slate-600 font-mono">Ep {p.episode}</span>
                    <div className="flex gap-2">
                      <span className="text-blue-600 font-semibold">{(p.winRate * 100).toFixed(0)}% W</span>
                      <span className="text-slate-500 font-semibold">{(p.drawRate * 100).toFixed(0)}% D</span>
                      <span className="text-rose-600 font-semibold">{(p.lossRate * 100).toFixed(0)}% L</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Hyperparameters */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-slate-800 font-bold pb-3 border-b-2 border-indigo-100">
            <div className="p-2 bg-indigo-100 rounded-lg">
              <Settings className="text-indigo-600" size={20} />
            </div>
            <h3 className="text-lg">Hyperparameters</h3>
          </div>
          
          <div className="space-y-4">
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-700 font-semibold">Learning Rate (α)</span>
                <span className="font-mono text-indigo-600 font-bold">{learningRate.toFixed(2)}</span>
              </div>
              <input 
                type="range" min="0.01" max="1" step="0.05"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full h-2 bg-indigo-200 rounded-lg appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #6366f1 ${learningRate * 100}%, #e0e7ff ${learningRate * 100}%)`
                }}
              />
              <p className="text-xs text-slate-500 mt-2">Controls how quickly the agent updates its knowledge</p>
            </div>

            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-700 font-semibold">Exploration (ε)</span>
                <span className="font-mono text-indigo-600 font-bold">{epsilon.toFixed(2)}</span>
              </div>
              <input 
                type="range" min="0" max="1" step="0.05"
                value={epsilon}
                onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                className="w-full h-2 bg-indigo-200 rounded-lg appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #6366f1 ${epsilon * 100}%, #e0e7ff ${epsilon * 100}%)`
                }}
              />
              <p className="text-xs text-slate-500 mt-2">
                {epsilon < 0.1 ? 'Mostly exploitation' : epsilon > 0.3 ? 'High exploration' : 'Balanced'}
              </p>
            </div>
          </div>
        </div>

        {/* Options */}
        <div className="space-y-3">
          <label className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-200 cursor-pointer hover:bg-slate-100 transition-all group">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-emerald-100 rounded-lg group-hover:bg-emerald-200 transition-colors">
                <Activity size={18} className="text-emerald-600" />
              </div>
              <div>
                <span className="text-sm font-semibold text-slate-700 block">Show Q-Values</span>
                <span className="text-xs text-slate-500">Display value estimates on board</span>
              </div>
            </div>
            <div className={`relative w-12 h-7 rounded-full transition-colors ${showQValues ? 'bg-emerald-500' : 'bg-slate-300'}`}>
              <div className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full shadow-md transform transition-transform ${showQValues ? 'translate-x-5' : ''}`} />
            </div>
            <input 
              type="checkbox" 
              className="hidden" 
              checked={showQValues} 
              onChange={() => setShowQValues(!showQValues)} 
            />
          </label>

          <label className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-200 cursor-pointer hover:bg-slate-100 transition-all group">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg group-hover:bg-blue-200 transition-colors">
                <Play size={18} className="text-blue-600" />
              </div>
              <div>
                <span className="text-sm font-semibold text-slate-700 block">Auto-Play</span>
                <span className="text-xs text-slate-500">Watch agent play continuously</span>
              </div>
            </div>
            <div className={`relative w-12 h-7 rounded-full transition-colors ${autoPlay ? 'bg-blue-500' : 'bg-slate-300'}`}>
              <div className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full shadow-md transform transition-transform ${autoPlay ? 'translate-x-5' : ''}`} />
            </div>
            <input 
              type="checkbox" 
              className="hidden" 
              checked={autoPlay} 
              onChange={() => { setAutoPlay(!autoPlay); if (!autoPlay) resetGame(); }} 
            />
          </label>
        </div>

        {/* Info Box */}
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-4 rounded-xl border border-blue-200">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Info size={18} className="text-blue-600" />
            </div>
            <div className="text-xs text-blue-900 leading-relaxed">
              <p className="font-semibold mb-1">Q-Learning Algorithm</p>
              <p className="text-blue-700">
                The agent learns optimal play by exploring different moves and updating its value estimates (Q-values) based on rewards. 
                Train the agent to improve its performance!
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * ============================================================================
 * MAIN APP
 * ============================================================================
 */

export default function RLStudio() {
  const [activeTab, setActiveTab] = useState('tictactoe');

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 font-sans text-slate-900 flex flex-col">
      <header className="bg-white/80 backdrop-blur-xl border-b border-slate-200/50 px-6 py-4 flex items-center justify-between sticky top-0 z-50 shadow-sm">
        <div className="flex items-center gap-4">
          <div className="relative">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 rounded-xl flex items-center justify-center text-white shadow-lg">
              <Brain size={28} />
            </div>
            <div className="absolute -top-1 -right-1 w-4 h-4 bg-emerald-500 rounded-full border-2 border-white animate-pulse" />
          </div>
          <div>
            <h1 className="text-2xl font-black bg-clip-text text-transparent bg-gradient-to-r from-blue-700 via-indigo-700 to-purple-700">
              RL Interactive Studio
            </h1>
            <p className="text-xs text-slate-600 font-semibold">Learn Reinforcement Learning by Playing</p>
          </div>
        </div>
        
        <nav className="flex bg-slate-100/80 backdrop-blur-sm p-1.5 rounded-xl gap-1 shadow-inner">
          <button 
            onClick={() => setActiveTab('tictactoe')}
            className={`px-6 py-2.5 rounded-lg text-sm font-bold transition-all ${
              activeTab === 'tictactoe' 
                ? 'bg-white text-blue-700 shadow-md scale-105' 
                : 'text-slate-600 hover:text-slate-900 hover:bg-white/50'
            }`}
          >
            Tic-Tac-Toe
          </button>
        </nav>
      </header>

      <main className="flex-1 overflow-auto p-4 md:p-8 max-w-[1800px] mx-auto w-full">
        <TicTacToe />
      </main>

      <footer className="bg-white/80 backdrop-blur-xl border-t border-slate-200/50 px-6 py-4 text-center">
        <p className="text-xs text-slate-500 font-medium">
          Built with React • Reinforcement Learning Interactive Tutorial
        </p>
      </footer>
    </div>
  );
}
