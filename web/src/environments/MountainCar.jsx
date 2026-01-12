import React, { useState, useEffect, useCallback } from 'react';
import { Mountain, Play, Pause, RotateCcw, Zap, Loader, ArrowLeft, ArrowRight, Target } from 'lucide-react';

const POSITION_MIN = -1.2;
const POSITION_MAX = 0.5;
const VELOCITY_MIN = -0.07;
const VELOCITY_MAX = 0.07;

const ACTIONS = [
  { id: 0, name: 'Reverse', icon: ArrowLeft, value: -1 },
  { id: 1, name: 'Neutral', value: 0 },
  { id: 2, name: 'Forward', icon: ArrowRight, value: 1 }
];

const MountainCar = ({ agent }) => {
  const [position, setPosition] = useState(-0.5);
  const [velocity, setVelocity] = useState(0);
  const [steps, setSteps] = useState(0);
  const [totalReward, setTotalReward] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [trajectory, setTrajectory] = useState([{position: -0.5, velocity: 0}]);
  const [trainingEpisodes, setTrainingEpisodes] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [currentAction, setCurrentAction] = useState(1); // Neutral

  const discretizePosition = (pos) => Math.floor((pos - POSITION_MIN) / ((POSITION_MAX - POSITION_MIN) / 20));
  const discretizeVelocity = (vel) => Math.floor((vel - VELOCITY_MIN) / ((VELOCITY_MAX - VELOCITY_MIN) / 20));
  
  const getStateKey = (pos, vel) => `${discretizePosition(pos)}_${discretizeVelocity(vel)}`;

  const step = useCallback((pos, vel, action) => {
    let newVel = vel + 0.001 * action - 0.0025 * Math.cos(3 * pos);
    newVel = Math.min(Math.max(VELOCITY_MIN, newVel), VELOCITY_MAX);
    let newPos = pos + newVel;
    newPos = Math.min(Math.max(POSITION_MIN, newPos), POSITION_MAX);
    if (newPos === POSITION_MIN) newVel = 0;
    return { position: newPos, velocity: newVel, reward: -1 };
  }, []);

  const resetCar = () => {
    const startPos = -0.5 + (Math.random() - 0.5) * 0.2;
    setPosition(startPos);
    setVelocity(0);
    setSteps(0);
    setTotalReward(0);
    setTrajectory([{position: startPos, velocity: 0}]);
    setIsRunning(false);
    setCurrentAction(1);
  };

  const trainAgent = async () => {
    setIsTraining(true);
    
    for (let ep = 0; ep < 100; ep++) { // Fewer episodes since Mountain Car is harder
      let pos = -0.5 + (Math.random() - 0.5) * 0.2;
      let vel = 0;
      let epSteps = 0;
      
      while (pos < POSITION_MAX && epSteps < 1000) {
        const stateKey = getStateKey(pos, vel);
        const action = agent.chooseAction(stateKey, [0, 1, 2]);
        const result = step(pos, vel, ACTIONS[action].value);
        
        const nextStateKey = getStateKey(result.position, result.velocity);
        const nextReward = result.position >= POSITION_MAX ? 100 : result.reward;
        agent.learn(stateKey, action, nextReward, nextStateKey, [0, 1, 2]);
        
        pos = result.position;
        vel = result.velocity;
        epSteps++;
        
        if (pos >= POSITION_MAX) break;
      }
    }
    
    setTrainingEpisodes(e => e + 100);
    setIsTraining(false);
  };

  const runStep = useCallback(() => {
    if (position >= POSITION_MAX || steps >= 1000) {
      setIsRunning(false);
      return;
    }

    const stateKey = getStateKey(position, velocity);
    const action = agent.chooseAction(stateKey, [0, 1, 2]);
    const result = step(position, velocity, ACTIONS[action].value);

    const nextStateKey = getStateKey(result.position, result.velocity);
    const nextReward = result.position >= POSITION_MAX ? 100 : result.reward;
    agent.learn(stateKey, action, nextReward, nextStateKey, [0, 1, 2]);

    setPosition(result.position);
    setVelocity(result.velocity);
    setTotalReward(prev => prev + nextReward);
    setSteps(prev => prev + 1);
    setTrajectory(prev => [...prev.slice(-100), {position: result.position, velocity: result.velocity}]);
    setCurrentAction(action);

    if (result.position >= POSITION_MAX) {
      setIsRunning(false);
    }
  }, [position, velocity, steps, agent]);

  useEffect(() => {
    let interval;
    if (isRunning && !isTraining) {
      interval = setInterval(runStep, 50);
    }
    return () => clearInterval(interval);
  }, [isRunning, runStep, isTraining]);

  const mountainPath = [];
  for (let x = POSITION_MIN; x <= POSITION_MAX; x += 0.05) {
    const y = Math.sin(3 * x) * 50 + 50;
    mountainPath.push({x: ((x - POSITION_MIN) / (POSITION_MAX - POSITION_MIN)) * 400, y: 100 - y});
  }

  const carX = ((position - POSITION_MIN) / (POSITION_MAX - POSITION_MIN)) * 400;
  const carY = 100 - (Math.sin(3 * position) * 50 + 50);

  return (
    <div className="flex flex-col xl:flex-row h-full gap-6">
      {/* Game Display */}
      <div className="flex-1 flex flex-col items-center justify-center bg-gradient-to-br from-white to-slate-50 rounded-2xl shadow-xl p-8 border border-slate-200">
        <div className="mb-6 text-center">
          <h2 className="text-3xl font-bold text-slate-800 flex items-center justify-center gap-3 mb-2">
            <Mountain className="text-green-600" size={32} /> Mountain Car
          </h2>
          <p className="text-slate-500 text-sm font-medium">Build momentum to reach the goal!</p>
        </div>

        {/* Mountain Visualization */}
        <div className="bg-white p-6 rounded-xl shadow-lg border-2 border-slate-200">
          <svg width="400" height="200" className="border border-slate-200 rounded-lg bg-gradient-to-b from-blue-50 to-white">
            {/* Mountain */}
            <path
              d={`M 0,${100} ${mountainPath.map(p => `L ${p.x},${p.y}`).join(' ')} L 400,100 Z`}
              fill="url(#mountainGradient)"
              stroke="#475569"
              strokeWidth="2"
            />
            <defs>
              <linearGradient id="mountainGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style={{stopColor: '#86efac', stopOpacity: 1}} />
                <stop offset="100%" style={{stopColor: '#22c55e', stopOpacity: 1}} />
              </linearGradient>
            </defs>
            
            {/* Goal flag */}
            <Target x={385} y={5} size={24} className="text-red-500" />
            <line x1="390" y1="15" x2="390" y2="30" stroke="#ef4444" strokeWidth="3" />
            
            {/* Car */}
            <circle cx={carX} cy={carY} r="8" fill="#3b82f6" stroke="#1e40af" strokeWidth="2" />
            
            {/* Velocity indicator */}
            <line 
              x1={carX} 
              y1={carY} 
              x2={carX + velocity * 500} 
              y2={carY}
              stroke={velocity > 0 ? '#10b981' : '#ef4444'}
              strokeWidth="2"
              markerEnd="url(#arrowhead)"
            />
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" fill={velocity > 0 ? '#10b981' : '#ef4444'} />
              </marker>
            </defs>
          </svg>
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
            <span className="text-xl font-black text-emerald-600">{position.toFixed(2)}</span>
            <span className="text-xs uppercase tracking-widest text-emerald-700 font-bold">Position</span>
          </div>
          <div className="flex flex-col items-center p-3 bg-amber-50 rounded-xl">
            <span className="text-xl font-black text-amber-600">{(velocity * 100).toFixed(1)}</span>
            <span className="text-xs uppercase tracking-widest text-amber-700 font-bold">Velocity</span>
          </div>
        </div>

        {/* Controls */}
        <div className="mt-6 flex gap-3">
          <button 
            onClick={() => setIsRunning(!isRunning)}
            disabled={position >= POSITION_MAX}
            className={`px-6 py-2 rounded-full font-medium shadow-lg transition-transform hover:scale-105 flex items-center gap-2 ${isRunning ? 'bg-amber-100 text-amber-700' : 'bg-slate-900 text-white'}`}
          >
            {isRunning ? <><Pause size={18} /> Pause</> : <><Play size={18} /> Start</>}
          </button>
          <button onClick={resetCar} className="px-6 py-2 bg-white border border-slate-200 text-slate-600 rounded-full font-medium hover:bg-slate-50">
            <RotateCcw size={18} className="inline mr-2" /> Reset
          </button>
        </div>
      </div>

      {/* Control Panel */}
      <div className="w-full xl:w-96 space-y-4">
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
              <span className="text-2xl font-black text-purple-700">{trainingEpisodes}</span>
            </div>
            <button 
              onClick={trainAgent}
              disabled={isTraining}
              className="w-full py-3 bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-lg font-bold shadow-lg hover:from-purple-700 hover:to-purple-800 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isTraining ? <><Loader className="animate-spin" size={16} /> Training...</> : <><Zap size={16} /> Train 100</>}
            </button>
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-5 rounded-xl border border-green-200">
          <div className="text-xs text-green-900">
            <p className="font-semibold mb-2">🏔️ Physics Challenge:</p>
            <p className="text-green-700">The car must rock back and forth to build momentum! It's underpowered and can't drive straight up. This is a classic delayed reward problem.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MountainCar;
