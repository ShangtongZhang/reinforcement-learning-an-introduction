import React, { useState, useEffect, useCallback } from 'react';
import { 
  Play, Pause, RotateCcw, Brain, Zap, Layout, 
  BarChart3, Settings, Info, Download, Upload, X, Check,
  Trophy, Book, Eye, EyeOff, Save, Loader, Award, Grid as GridIcon, Mountain
} from 'lucide-react';
import GridWorld from './environments/GridWorld';
import CliffWalking from './environments/CliffWalking';
import MountainCar from './environments/MountainCar';

/**
 * RL AGENT CLASS
 */
class RLAgent {
  constructor(alpha = 0.5, gamma = 0.9, epsilon = 0.1) {
    this.qTable = {};
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this.visitCounts = {};
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

  learn(state, action, reward, nextState, nextAvailableActions) {
    const currentQ = this.getQ(state, action);
    const maxNextQ = nextAvailableActions && nextAvailableActions.length > 0
      ? Math.max(...nextAvailableActions.map(a => this.getQ(nextState, a)))
      : 0;
    const target = reward + this.gamma * maxNextQ;
    const newQ = currentQ + this.alpha * (target - currentQ);
    this.setQ(state, action, newQ);
    this.incrementVisit(state);
  }

  exportPolicy() {
    return {
      qTable: this.qTable,
      visitCounts: this.visitCounts,
      config: { alpha: this.alpha, gamma: this.gamma, epsilon: this.epsilon },
      timestamp: new Date().toISOString()
    };
  }

  importPolicy(data) {
    this.qTable = data.qTable || {};
    this.visitCounts = data.visitCounts || {};
    if (data.config) {
      this.alpha = data.config.alpha;
      this.gamma = data.config.gamma;
      this.epsilon = data.config.epsilon;
    }
  }
}

/**
 * TUTORIAL MODAL
 */
const Tutorial = ({ onClose }) => {
  const [step, setStep] = useState(0);

  const steps = [
    {
      title: "Welcome to Q-Learning!",
      content: "You're about to learn how AI agents learn through trial and error. The agent learns to play Tic-Tac-Toe by playing many games and improving its strategy."
    },
    {
      title: "How Q-Learning Works",
      content: "The agent maintains a 'Q-table' that stores values for each state-action pair. Higher values mean better moves. It updates these values based on whether it won or lost."
    },
    {
      title: "Training the Agent",
      content: "Click 'Train 1,000 Episodes' to let the agent play 1,000 games against a random opponent. Each game helps it learn which moves lead to wins!"
    },
    {
      title: "Playing the Game",
      content: "Click any empty cell to place your X. The agent will respond with O. After training, enable 'Show Q-Values' to see what the agent has learned!"
    }
  ];

  const currentStep = steps[step];

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full p-8 relative">
        <button onClick={onClose} className="absolute top-4 right-4 p-2 hover:bg-slate-100 rounded-lg">
          <X size={24} />
        </button>

        <div className="flex flex-col items-center text-center">
          <Book className="text-blue-600 mb-4" size={48} />
          <h2 className="text-3xl font-black text-slate-900 mb-4">{currentStep.title}</h2>
          <p className="text-lg text-slate-600 leading-relaxed mb-8">{currentStep.content}</p>

          <div className="flex items-center gap-2 mb-6">
            {steps.map((_, idx) => (
              <div key={idx} className={`h-2 rounded-full transition-all ${idx === step ? 'w-8 bg-blue-600' : 'w-2 bg-slate-300'}`} />
            ))}
          </div>

          <div className="flex gap-3">
            {step > 0 && (
              <button onClick={() => setStep(step - 1)} className="px-6 py-3 bg-slate-100 text-slate-700 rounded-xl font-semibold hover:bg-slate-200">
                Previous
              </button>
            )}
            {step < steps.length - 1 ? (
              <button onClick={() => setStep(step + 1)} className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-semibold hover:shadow-lg">
                Next
              </button>
            ) : (
              <button onClick={onClose} className="px-6 py-3 bg-gradient-to-r from-emerald-600 to-green-600 text-white rounded-xl font-semibold hover:shadow-lg">
                Start Learning!
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * EXPORT/IMPORT MODAL
 */
const ExportImportModal = ({ agent, onClose, onImport }) => {
  const [mode, setMode] = useState('export');
  const [importData, setImportData] = useState('');
  const [message, setMessage] = useState('');

  const handleExport = () => {
    const data = agent.exportPolicy();
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rl-agent-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setMessage('✓ Policy exported!');
    setTimeout(() => setMessage(''), 3000);
  };

  const handleCopyToClipboard = () => {
    const data = agent.exportPolicy();
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setMessage('✓ Copied to clipboard!');
    setTimeout(() => setMessage(''), 3000);
  };

  const handleImport = () => {
    try {
      const data = JSON.parse(importData);
      onImport(data);
      setMessage('✓ Policy imported!');
      setTimeout(() => { setMessage(''); onClose(); }, 1500);
    } catch (error) {
      setMessage('✗ Error: Invalid JSON');
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => setImportData(event.target.result);
      reader.readAsText(file);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full p-8 relative">
        <button onClick={onClose} className="absolute top-4 right-4 p-2 hover:bg-slate-100 rounded-lg">
          <X size={24} />
        </button>

        <h2 className="text-2xl font-black text-slate-900 mb-6">Save & Load Agent</h2>

        <div className="flex gap-2 mb-6">
          <button onClick={() => setMode('export')} className={`flex-1 py-2 rounded-lg font-semibold ${mode === 'export' ? 'bg-blue-100 text-blue-700' : 'bg-slate-100'}`}>
            <Download size={18} className="inline mr-2" /> Export
          </button>
          <button onClick={() => setMode('import')} className={`flex-1 py-2 rounded-lg font-semibold ${mode === 'import' ? 'bg-blue-100 text-blue-700' : 'bg-slate-100'}`}>
            <Upload size={18} className="inline mr-2" /> Import
          </button>
        </div>

        {mode === 'export' ? (
          <div className="space-y-4">
            <p className="text-sm text-slate-600 mb-4">Export the trained agent's Q-table to save your progress.</p>
            <div className="flex gap-2">
              <button onClick={handleExport} className="flex-1 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-semibold">
                <Download size={18} className="inline mr-2" /> Download
              </button>
              <button onClick={handleCopyToClipboard} className="flex-1 py-3 bg-slate-700 text-white rounded-lg font-semibold">
                <Save size={18} className="inline mr-2" /> Copy
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <label className="block w-full py-4 bg-white border-2 border-dashed border-slate-300 rounded-lg text-center cursor-pointer hover:border-blue-400">
              <Upload size={20} className="inline mr-2" />
              <span className="font-semibold">Choose File</span>
              <input type="file" accept=".json" onChange={handleFileUpload} className="hidden" />
            </label>
            <textarea
              value={importData}
              onChange={(e) => setImportData(e.target.value)}
              placeholder='{"qTable": {...}}'
              className="w-full h-32 p-3 border rounded-lg font-mono text-xs"
            />
            <button onClick={handleImport} disabled={!importData} className="w-full py-3 bg-emerald-600 text-white rounded-lg font-semibold disabled:opacity-50">
              <Check size={18} className="inline mr-2" /> Import
            </button>
          </div>
        )}

        {message && (
          <div className={`mt-4 p-3 rounded-lg text-center font-semibold ${message.includes('✗') ? 'bg-rose-100 text-rose-700' : 'bg-emerald-100 text-emerald-700'}`}>
            {message}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * TIC-TAC-TOE GAME
 */
const TTT_WIN_COMBOS = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6]
];

const checkWinner = (board) => {
  for (let combo of TTT_WIN_COMBOS) {
    const [a, b, c] = combo;
    if (board[a] && board[a] === board[b] && board[a] === board[c]) return board[a];
  }
  return !board.includes(null) ? 'DRAW' : null;
};

const TicTacToe = ({ agent, showTutorial }) => {
  const [board, setBoard] = useState(Array(9).fill(null));
  const [isXNext, setIsXNext] = useState(true);
  const [gameStatus, setGameStatus] = useState('playing');
  const [winner, setWinner] = useState(null);
  const [trainingEpisodes, setTrainingEpisodes] = useState(0);
  const [showQValues, setShowQValues] = useState(false);
  const [autoPlay, setAutoPlay] = useState(false);
  const [epsilon, setEpsilon] = useState(0.15);
  const [learningRate, setLearningRate] = useState(0.5);
  const [stats, setStats] = useState({ wins: 0, losses: 0, draws: 0, totalGames: 0 });
  const [isTraining, setIsTraining] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);

  useEffect(() => {
    agent.epsilon = epsilon;
    agent.alpha = learningRate;
  }, [epsilon, learningRate, agent]);

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
      const timer = setTimeout(() => makeBotMove(), autoPlay ? 50 : 600);
      return () => clearTimeout(timer);
    }
  }, [isXNext, gameStatus, autoPlay, isTraining, board]);

  const makeBotMove = () => {
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
  };

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

  const trainAgent = async () => {
    setIsTraining(true);
    
    for (let i = 0; i < 1000; i++) {
      let b = Array(9).fill(null);
      let turn = Math.random() > 0.5 ? 'X' : 'O';
      
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
          let reward = res === 'O' ? 10 : (res === 'X' ? -10 : (res === 'DRAW' ? 5 : -0.1));
          agent.learn(stateStr, action, reward, getBoardState(b), getAvailableMoves(b));
        }
        
        if (res) break;
        turn = turn === 'X' ? 'O' : 'X';
      }
    }

    setTrainingEpisodes(e => e + 1000);
    setIsTraining(false);
  };

  const winRate = stats.totalGames > 0 ? (stats.wins / stats.totalGames * 100).toFixed(1) : 0;

  return (
    <div className="flex flex-col xl:flex-row h-full gap-6">
      {showExportModal && (
        <ExportImportModal 
          agent={agent}
          onClose={() => setShowExportModal(false)}
          onImport={(data) => {
            agent.importPolicy(data);
            setTrainingEpisodes(Object.keys(agent.qTable).length);
          }}
        />
      )}

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
                  className={`relative w-24 h-24 rounded-xl flex items-center justify-center text-5xl font-black transition-all duration-200 shadow-md hover:shadow-lg hover:scale-105 disabled:hover:scale-100 ${bgClass}`}
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
              <button onClick={resetGame} className="px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-full font-bold shadow-lg hover:shadow-xl hover:scale-105 transition-all flex items-center gap-2">
                <RotateCcw size={20} /> Play Again
              </button>
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="mt-8 grid grid-cols-4 gap-3 w-full max-w-md">
          {[
            { label: 'Wins', value: stats.wins, bg: 'bg-blue-50', text: 'text-blue-600' },
            { label: 'Draws', value: stats.draws, bg: 'bg-slate-50', text: 'text-slate-600' },
            { label: 'Losses', value: stats.losses, bg: 'bg-rose-50', text: 'text-rose-600' },
            { label: 'Total', value: stats.totalGames, bg: 'bg-purple-50', text: 'text-purple-600' }
          ].map(({ label, value, bg, text }) => (
            <div key={label} className={`flex flex-col items-center p-4 ${bg} rounded-xl shadow-sm`}>
              <span className={`text-3xl font-black ${text}`}>{value}</span>
              <span className="text-xs uppercase tracking-widest text-slate-700 font-bold">{label}</span>
            </div>
          ))}
        </div>

        {stats.totalGames > 0 && (
          <div className="mt-6 w-full max-w-md bg-white p-6 rounded-xl shadow-lg border border-slate-200">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-bold text-slate-700 flex items-center gap-2">
                <BarChart3 size={18} className="text-blue-600" /> Performance
              </span>
              <span className="text-2xl font-black text-blue-600">{winRate}%</span>
            </div>
            <div className="flex gap-1 h-3 rounded-full overflow-hidden bg-slate-100">
              <div className="bg-blue-500 transition-all duration-500" style={{ width: `${(stats.wins / stats.totalGames) * 100}%` }} />
              <div className="bg-slate-400 transition-all duration-500" style={{ width: `${(stats.draws / stats.totalGames) * 100}%` }} />
              <div className="bg-rose-500 transition-all duration-500" style={{ width: `${(stats.losses / stats.totalGames) * 100}%` }} />
            </div>
          </div>
        )}
      </div>

      {/* Control Panel */}
      <div className="w-full xl:w-96 space-y-4">
        {/* Training */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-slate-200">
          <div className="flex items-center gap-2 mb-4 pb-3 border-b-2 border-purple-100">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Brain size={20} className="text-purple-600" />
            </div>
            <h3 className="font-bold text-lg">Agent Brain</h3>
          </div>
          
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-5 rounded-xl border border-purple-200">
            <div className="flex justify-between items-center mb-3">
              <span className="text-xs font-bold text-purple-700 uppercase">Episodes</span>
              <span className="text-2xl font-black text-purple-700">{trainingEpisodes.toLocaleString()}</span>
            </div>
            <button 
              onClick={trainAgent}
              disabled={isTraining}
              className="w-full py-3 bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-lg font-bold shadow-lg hover:from-purple-700 hover:to-purple-800 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isTraining ? <><Loader className="animate-spin" size={16} /> Training...</> : <><Zap size={16} /> Train 1,000</>}
            </button>
          </div>
        </div>

        {/* Parameters */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-slate-200">
          <div className="flex items-center gap-2 mb-4 pb-3 border-b-2 border-indigo-100">
            <div className="p-2 bg-indigo-100 rounded-lg">
              <Settings size={20} className="text-indigo-600" />
            </div>
            <h3 className="font-bold text-lg">Parameters</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="font-semibold">Learning Rate (α)</span>
                <span className="font-mono text-indigo-600 font-bold">{learningRate.toFixed(2)}</span>
              </div>
              <input type="range" min="0.01" max="1" step="0.05" value={learningRate} onChange={(e) => setLearningRate(parseFloat(e.target.value))} className="w-full" />
            </div>

            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="font-semibold">Exploration (ε)</span>
                <span className="font-mono text-indigo-600 font-bold">{epsilon.toFixed(2)}</span>
              </div>
              <input type="range" min="0" max="1" step="0.05" value={epsilon} onChange={(e) => setEpsilon(parseFloat(e.target.value))} className="w-full" />
            </div>
          </div>
        </div>

        {/* Save/Load */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-slate-200">
          <button onClick={() => setShowExportModal(true)} className="w-full py-3 bg-gradient-to-r from-emerald-600 to-green-600 text-white rounded-lg font-bold shadow-lg hover:shadow-xl transition-all flex items-center justify-center gap-2">
            <Save size={18} /> Save/Load Agent
          </button>
        </div>

        {/* Options */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-slate-200 space-y-3">
          <label className="flex items-center justify-between p-3 bg-slate-50 rounded-lg cursor-pointer hover:bg-slate-100">
            <span className="text-sm font-semibold flex items-center gap-2">
              {showQValues ? <Eye size={18} /> : <EyeOff size={18} />} Show Q-Values
            </span>
            <input type="checkbox" checked={showQValues} onChange={() => setShowQValues(!showQValues)} className="w-4 h-4" />
          </label>

          <label className="flex items-center justify-between p-3 bg-slate-50 rounded-lg cursor-pointer hover:bg-slate-100">
            <span className="text-sm font-semibold flex items-center gap-2">
              {autoPlay ? <Pause size={18} /> : <Play size={18} />} Auto-Play
            </span>
            <input type="checkbox" checked={autoPlay} onChange={() => { setAutoPlay(!autoPlay); if (!autoPlay) resetGame(); }} className="w-4 h-4" />
          </label>

          <button onClick={showTutorial} className="w-full py-3 bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-700 rounded-lg font-semibold hover:shadow-md transition-all flex items-center justify-center gap-2">
            <Book size={18} /> Tutorial
          </button>
        </div>

        {/* Info */}
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-5 rounded-xl border border-blue-200">
          <div className="flex items-start gap-3">
            <Info size={20} className="text-blue-600" />
            <div className="text-xs text-blue-900">
              <p className="font-semibold mb-2">Q-Learning Algorithm</p>
              <p className="text-blue-700">The agent learns optimal play by exploring moves and updating Q-values based on rewards!</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * MAIN APP
 */
export default function RLStudio() {
  const [showTutorial, setShowTutorial] = useState(false);
  const [activeTab, setActiveTab] = useState('tictactoe');
  const [agent] = useState(() => new RLAgent(0.5, 0.9, 0.15));

  useEffect(() => {
    const hasSeenTutorial = localStorage.getItem('rl-studio-tutorial-seen');
    if (!hasSeenTutorial) {
      setShowTutorial(true);
      localStorage.setItem('rl-studio-tutorial-seen', 'true');
    }
  }, []);

  const tabs = [
    { id: 'tictactoe', name: 'Tic-Tac-Toe', icon: Layout },
    { id: 'gridworld', name: 'Grid World', icon: GridIcon },
    { id: 'cliffwalking', name: 'Cliff Walking', icon: Mountain },
    { id: 'mountaincar', name: 'Mountain Car', icon: Mountain }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 font-sans text-slate-900 flex flex-col">
      {showTutorial && <Tutorial onClose={() => setShowTutorial(false)} />}

      <header className="bg-white/80 backdrop-blur-xl border-b border-slate-200/50 px-6 py-4 flex flex-col md:flex-row items-center justify-between gap-4 sticky top-0 z-40 shadow-sm">
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
        
        <div className="flex items-center gap-3">
          <nav className="flex bg-slate-100 p-1 rounded-xl gap-1">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all flex items-center gap-2 ${
                  activeTab === tab.id 
                    ? 'bg-white text-blue-700 shadow-md' 
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {React.createElement(tab.icon, { size: 16 })} {tab.name}
              </button>
            ))}
          </nav>
          <button onClick={() => setShowTutorial(true)} className="px-4 py-2 bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-700 rounded-lg font-semibold hover:shadow-md transition-all flex items-center gap-2">
            <Book size={18} /> Tutorial
          </button>
        </div>
      </header>

      <main className="flex-1 overflow-auto p-4 md:p-8 max-w-[1800px] mx-auto w-full">
        {activeTab === 'tictactoe' && <TicTacToe agent={agent} showTutorial={() => setShowTutorial(true)} />}
        {activeTab === 'gridworld' && <GridWorld agent={agent} />}
        {activeTab === 'cliffwalking' && <CliffWalking agent={agent} />}
        {activeTab === 'mountaincar' && <MountainCar agent={agent} />}
      </main>

      <footer className="bg-white/80 backdrop-blur-xl border-t border-slate-200/50 px-6 py-4 text-center">
        <p className="text-xs text-slate-500 font-medium">
          Built with React • Reinforcement Learning Interactive Tutorial
        </p>
      </footer>
    </div>
  );
}
