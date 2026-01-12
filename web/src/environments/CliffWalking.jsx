import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Mountain, Target, AlertTriangle, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Play, Pause, RotateCcw, Maximize2, Loader, Zap } from 'lucide-react';

const WORLD_HEIGHT = 4;
const WORLD_WIDTH = 12;
const START = {x: 0, y: 3};
const GOAL = {x: 11, y: 3};

const ACTIONS = [
  {id: 0, name: 'Up', icon: ArrowUp, dx: 0, dy: -1},
  {id: 1, name: 'Down', icon: ArrowDown, dx: 0, dy: 1},
  {id: 2, name: 'Left', icon: ArrowLeft, dx: -1, dy: 0},
  {id: 3, name: 'Right', icon: ArrowRight, dx: 1, dy: 0}
];

const isCliff = (x, y) => y === 2 && x >= 1 && x <= 10;

const CliffWalking = ({ agent }) => {
  const [agentPos, setAgentPos] = useState({...START});
  const [isRunning, setIsRunning] = useState(false);
  const [path, setPath] = useState([{...START}]);
  const [totalReward, setTotalReward] = useState(0);
  const [steps, setSteps] = useState(0);
  const [trainingEpisodes, setTrainingEpisodes] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [showValueHeatmap, setShowValueHeatmap] = useState(false);
  const [showPolicyHeatmap, setShowPolicyHeatmap] = useState(false);
  const [autoPlay, setAutoPlay] = useState(false);
  const [episodeCount, setEpisodeCount] = useState(0);

  const getStateKey = (pos) => `${pos.x}_${pos.y}`;
  const isGoal = (x, y) => x === GOAL.x && y === GOAL.y;

  const getAvailableActions = (pos) => {
    return ACTIONS.filter(action => {
      const nx = pos.x + action.dx;
      const ny = pos.y + action.dy;
      return nx >= 0 && nx < WORLD_WIDTH && ny >= 0 && ny < WORLD_HEIGHT;
    });
  };

  const step = useCallback((pos, action) => {
    const newPos = {x: pos.x + action.dx, y: pos.y + action.dy};
    if (newPos.x < 0 || newPos.x >= WORLD_WIDTH || newPos.y < 0 || newPos.y >= WORLD_HEIGHT) {
      return pos;
    }
    return newPos;
  }, []);

  const getReward = (pos, fellOffCliff) => {
    if (isGoal(pos.x, pos.y)) return 0;
    if (fellOffCliff) return -100;
    return -1;
  };

  const resetGrid = () => {
    setAgentPos({...START});
    setPath([{...START}]);
    setTotalReward(0);
    setSteps(0);
    setIsRunning(false);
  };

  const trainAgent = async () => {
    setIsTraining(true);
    
    for (let episode = 0; episode < 1000; episode++) {
      let pos = {...START};
      let episodeSteps = 0;
      
      while (!isGoal(pos.x, pos.y) && episodeSteps < 500) {
        const stateKey = getStateKey(pos);
        const available = getAvailableActions(pos);
        if (available.length === 0) break;
        
        const action = agent.chooseAction(stateKey, available.map(a => a.id));
        const actionObj = ACTIONS.find(a => a.id === action);
        const nextPos = step(pos, actionObj);
        
        let fellOffCliff = false;
        if (isCliff(nextPos.x, nextPos.y)) {
          fellOffCliff = true;
          nextPos.x = START.x;
          nextPos.y = START.y;
        }
        
        const reward = getReward(nextPos, fellOffCliff);
        const nextStateKey = getStateKey(nextPos);
        const nextAvailable = getAvailableActions(nextPos);
        agent.learn(stateKey, action, reward, nextStateKey, nextAvailable.map(a => a.id));
        
        pos = nextPos;
        episodeSteps++;
        
        if (isGoal(pos.x, pos.y)) break;
      }
    }
    
    setTrainingEpisodes(e => e + 1000);
    setIsTraining(false);
  };

  const runEpisode = useCallback(() => {
    if (isGoal(agentPos.x, agentPos.y)) {
      setIsRunning(false);
      setEpisodeCount(prev => prev + 1);
      if (autoPlay) {
        setTimeout(() => resetGrid(), 500);
      }
      return;
    }

    const stateKey = getStateKey(agentPos);
    const available = getAvailableActions(agentPos);
    if (available.length === 0) {
      setIsRunning(false);
      return;
    }

    const action = agent.chooseAction(stateKey, available.map(a => a.id));
    const actionObj = ACTIONS.find(a => a.id === action);
    let nextPos = step(agentPos, actionObj);
    
    let fellOffCliff = false;
    if (isCliff(nextPos.x, nextPos.y)) {
      fellOffCliff = true;
      nextPos = {...START};
    }
    
    const reward = getReward(nextPos, fellOffCliff);
    const nextStateKey = getStateKey(nextPos);
    const nextAvailable = getAvailableActions(nextPos);
    agent.learn(stateKey, action, reward, nextStateKey, nextAvailable.map(a => a.id));

    setAgentPos(nextPos);
    setPath(prev => [...prev, {...nextPos}]);
    setTotalReward(prev => prev + reward);
    setSteps(prev => prev + 1);

    if (isGoal(nextPos.x, nextPos.y) || steps >= 500) {
      setIsRunning(false);
      setEpisodeCount(prev => prev + 1);
      if (autoPlay) {
        setTimeout(() => resetGrid(), 500);
      }
    }
  }, [agentPos, agent, steps, autoPlay]);

  useEffect(() => {
    let interval;
    if (isRunning && !isTraining) {
      interval = setInterval(() => {
        runEpisode();
      }, autoPlay ? 50 : 200);
    }
    return () => clearInterval(interval);
  }, [isRunning, runEpisode, isTraining, autoPlay]);

  // Value heatmap
  const valueHeatmap = useMemo(() => {
    const values = [];
    for (let y = 0; y < WORLD_HEIGHT; y++) {
      const row = [];
      for (let x = 0; x < WORLD_WIDTH; x++) {
        const stateKey = getStateKey({x, y});
        const available = getAvailableActions({x, y});
        if (available.length === 0) {
          row.push(null);
        } else {
          const maxQ = Math.max(...available.map(a => agent.getQ(stateKey, a.id)));
          row.push(maxQ);
        }
      }
      values.push(row);
    }
    return values;
  }, [agent, trainingEpisodes]);

  // Policy heatmap
  const policyHeatmap = useMemo(() => {
    const policy = [];
    for (let y = 0; y < WORLD_HEIGHT; y++) {
      const row = [];
      for (let x = 0; x < WORLD_WIDTH; x++) {
        const stateKey = getStateKey({x, y});
        const available = getAvailableActions({x, y});
        if (available.length === 0) {
          row.push(null);
        } else {
          const bestAction = agent.getBestAction(stateKey, available.map(a => a.id));
          row.push(bestAction);
        }
      }
      policy.push(row);
    }
    return policy;
  }, [agent, trainingEpisodes]);

  const maxValue = Math.max(...valueHeatmap.flat().filter(v => v !== null), 0);
  const minValue = Math.min(...valueHeatmap.flat().filter(v => v !== null), 0);

  const getValueColor = (value) => {
    if (value === null) return 'bg-slate-800';
    const normalized = (value - minValue) / (maxValue - minValue || 1);
    const hue = normalized * 240;
    return `hsl(${240 - hue}, 70%, 50%)`;
  };

  return (
    <div className="flex flex-col xl:flex-row h-full gap-6">
      {/* Grid Display */}
      <div className="flex-1 flex flex-col items-center justify-center bg-gradient-to-br from-white to-slate-50 rounded-2xl shadow-xl p-8 border border-slate-200">
        <div className="mb-6 text-center">
          <h2 className="text-3xl font-bold text-slate-800 flex items-center justify-center gap-3 mb-2">
            <Mountain className="text-rose-600" size={32} /> Cliff Walking
          </h2>
          <p className="text-slate-500 text-sm font-medium">Navigate safely to the goal!</p>
        </div>

        <div className="relative">
          <div className="grid grid-cols-12 gap-0.5 bg-slate-200 p-2 rounded-lg border-2 border-slate-300">
            {Array.from({ length: WORLD_HEIGHT * WORLD_WIDTH }).map((_, i) => {
              const x = i % WORLD_WIDTH;
              const y = Math.floor(i / WORLD_WIDTH);
              
              const isAgent = agentPos.x === x && agentPos.y === y;
              const isG = isGoal(x, y);
              const isClf = isCliff(x, y);
              const isSt = x === START.x && y === START.y;
              const isVisited = path.some(p => p.x === x && p.y === y) && !isAgent;
              const value = valueHeatmap[y][x];
              const policyAction = policyHeatmap[y][x];

              let cellClass = "bg-white";
              if (isG) cellClass = "bg-green-200";
              if (isClf) cellClass = "bg-red-300";
              if (showValueHeatmap && value !== null) {
                cellClass = getValueColor(value);
              }

              return (
                <div key={i} className={`w-10 h-10 rounded flex items-center justify-center relative ${cellClass} border ${isAgent ? 'border-blue-500 border-2' : 'border-transparent'}`}>
                  {isG && !isAgent && <Target size={16} className="text-green-700" />}
                  {isClf && !isAgent && <AlertTriangle size={14} className="text-red-700 opacity-70" />}
                  {isVisited && !isClf && !isG && <div className="w-1.5 h-1.5 rounded-full bg-blue-200" />}
                  {isAgent && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-8 h-8 bg-blue-600 rounded-full shadow-lg flex items-center justify-center text-white text-[10px] font-bold">
                        A
                      </div>
                    </div>
                  )}
                  {showPolicyHeatmap && policyAction !== null && !isAgent && !isG && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      {React.createElement(ACTIONS.find(a => a.id === policyAction)?.icon || ArrowUp, {size: 14, className: "text-slate-700 opacity-50"})}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Stats */}
        <div className="mt-8 grid grid-cols-4 gap-3 w-full max-w-lg">
          <div className="flex flex-col items-center p-3 bg-blue-50 rounded-xl">
            <span className="text-xl font-black text-blue-600">{steps}</span>
            <span className="text-xs uppercase tracking-widest text-blue-700 font-bold">Steps</span>
          </div>
          <div className="flex flex-col items-center p-3 bg-purple-50 rounded-xl">
            <span className="text-xl font-black text-purple-600">{totalReward}</span>
            <span className="text-xs uppercase tracking-widest text-purple-700 font-bold">Reward</span>
          </div>
          <div className="flex flex-col items-center p-3 bg-emerald-50 rounded-xl">
            <span className="text-xl font-black text-emerald-600">{episodeCount}</span>
            <span className="text-xs uppercase tracking-widest text-emerald-700 font-bold">Episodes</span>
          </div>
          <div className="flex flex-col items-center p-3 bg-amber-50 rounded-xl">
            <span className="text-xl font-black text-amber-600">{trainingEpisodes.toLocaleString()}</span>
            <span className="text-xs uppercase tracking-widest text-amber-700 font-bold">Trained</span>
          </div>
        </div>

        {/* Controls */}
        <div className="mt-6 flex gap-3">
          <button 
            onClick={() => setIsRunning(!isRunning)}
            disabled={isGoal(agentPos.x, agentPos.y)}
            className={`px-6 py-2 rounded-full font-medium shadow-lg transition-transform hover:scale-105 flex items-center gap-2 ${isRunning ? 'bg-amber-100 text-amber-700' : 'bg-slate-900 text-white'}`}
          >
            {isRunning ? <><Pause size={18} /> Pause</> : <><Play size={18} /> Start</>}
          </button>
          <button 
            onClick={resetGrid}
            className="px-6 py-2 bg-white border border-slate-200 text-slate-600 rounded-full font-medium hover:bg-slate-50"
          >
            <RotateCcw size={18} className="inline mr-2" /> Reset
          </button>
        </div>
      </div>

      {/* Control Panel */}
      <div className="w-full xl:w-96 space-y-4">
        {/* Training */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-slate-200">
          <div className="flex items-center gap-2 mb-4 pb-3 border-b-2 border-purple-100">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Mountain size={20} className="text-purple-600" />
            </div>
            <h3 className="font-bold text-lg">Training</h3>
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

        {/* Visualizations */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-slate-200 space-y-3">
          <label className="flex items-center justify-between p-3 bg-slate-50 rounded-lg cursor-pointer hover:bg-slate-100">
            <span className="text-sm font-semibold flex items-center gap-2">
              <Maximize2 size={18} /> Value Heatmap
            </span>
            <input type="checkbox" checked={showValueHeatmap} onChange={() => setShowValueHeatmap(!showValueHeatmap)} className="w-4 h-4" />
          </label>

          <label className="flex items-center justify-between p-3 bg-slate-50 rounded-lg cursor-pointer hover:bg-slate-100">
            <span className="text-sm font-semibold flex items-center gap-2">
              <Maximize2 size={18} /> Policy Heatmap
            </span>
            <input type="checkbox" checked={showPolicyHeatmap} onChange={() => setShowPolicyHeatmap(!showPolicyHeatmap)} className="w-4 h-4" />
          </label>

          <label className="flex items-center justify-between p-3 bg-slate-50 rounded-lg cursor-pointer hover:bg-slate-100">
            <span className="text-sm font-semibold flex items-center gap-2">
              {autoPlay ? <Pause size={18} /> : <Play size={18} />} Auto-Play
            </span>
            <input type="checkbox" checked={autoPlay} onChange={() => { setAutoPlay(!autoPlay); if (!autoPlay) resetGrid(); }} className="w-4 h-4" />
          </label>
        </div>

        {/* Info */}
        <div className="bg-gradient-to-br from-rose-50 to-red-50 p-5 rounded-xl border border-rose-200">
          <div className="text-xs text-rose-900">
            <p className="font-semibold mb-2">⚠️ Cliff Warning:</p>
            <p className="text-rose-700">Falling off the cliff (red area) resets you to start with -100 reward! The agent learns to take the safe path around the top.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CliffWalking;
