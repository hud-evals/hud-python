import React, { useState } from 'react';

interface GameState {
  score: number;
  moves: number;
  game_over: boolean;
  won: boolean;
  highest_tile: number;
  target_tile: number;
}

interface GameControlsProps {
  gameState: GameState;
  onNewGame: (boardSize: number, targetTile: number) => void;
  message: string;
}

export default function GameControls({ gameState, onNewGame, message }: GameControlsProps) {
  const [targetTile, setTargetTile] = useState(gameState.target_tile);
  const [boardSize, setBoardSize] = useState(4);

  const efficiency = gameState.moves > 0 
    ? (gameState.score / gameState.moves).toFixed(1)
    : '0.0';

  return (
    <div className="mb-6">
      {/* Score and Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="bg-white rounded-lg p-3 shadow">
          <div className="text-sm text-gray-600">Score</div>
          <div className="text-2xl font-bold text-gray-800">{gameState.score}</div>
        </div>
        <div className="bg-white rounded-lg p-3 shadow">
          <div className="text-sm text-gray-600">Moves</div>
          <div className="text-2xl font-bold text-gray-800">{gameState.moves}</div>
        </div>
        <div className="bg-white rounded-lg p-3 shadow">
          <div className="text-sm text-gray-600">Highest</div>
          <div className="text-2xl font-bold text-gray-800">{gameState.highest_tile}</div>
        </div>
        <div className="bg-white rounded-lg p-3 shadow">
          <div className="text-sm text-gray-600">Efficiency</div>
          <div className="text-2xl font-bold text-gray-800">{efficiency}</div>
        </div>
      </div>

      {/* Game Controls */}
      <div className="bg-white rounded-lg p-4 shadow mb-4">
        <div className="flex flex-col sm:flex-row gap-4 items-center">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Target:</label>
            <select
              value={targetTile}
              onChange={(e) => setTargetTile(Number(e.target.value))}
              className="px-2 py-1 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={64}>64</option>
              <option value={128}>128</option>
              <option value={256}>256</option>
              <option value={512}>512</option>
              <option value={1024}>1024</option>
              <option value={2048}>2048</option>
              <option value={4096}>4096</option>
              <option value={8192}>8192</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Size:</label>
            <select
              value={boardSize}
              onChange={(e) => setBoardSize(Number(e.target.value))}
              className="px-2 py-1 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={3}>3x3</option>
              <option value={4}>4x4</option>
              <option value={5}>5x5</option>
              <option value={6}>6x6</option>
            </select>
          </div>

          <button
            onClick={() => onNewGame(boardSize, targetTile)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          >
            New Game
          </button>
        </div>
      </div>

      {/* Status Message */}
      {message && (
        <div className={`text-center p-3 rounded-lg ${
          gameState.won ? 'bg-green-100 text-green-800' : 
          gameState.game_over ? 'bg-red-100 text-red-800' : 
          'bg-blue-100 text-blue-800'
        }`}>
          {message}
        </div>
      )}
    </div>
  );
}