'use client';

import { useState, useEffect, useCallback } from 'react';
import GameBoard from '../components/GameBoard';
import GameControls from '../components/GameControls';

// Dynamically determine API URL based on current port
// Backend is always on frontend_port + 1
const getApiUrl = () => {
  if (typeof window !== 'undefined') {
    const currentPort = parseInt(window.location.port) || 3000;
    return `http://localhost:${currentPort + 1}`;
  }
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';
};

const API_URL = getApiUrl();

interface GameState {
  board: number[][];
  score: number;
  moves: number;
  game_over: boolean;
  won: boolean;
  highest_tile: number;
  target_tile: number;
  board_size: number;
}

export default function Game2048() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  // Load initial game state
  useEffect(() => {
    fetchGameState();
  }, []);

  // Handle keyboard input
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (gameState?.game_over) return;
      
      const keyMap: { [key: string]: string } = {
        'ArrowUp': 'up',
        'ArrowDown': 'down',
        'ArrowLeft': 'left',
        'ArrowRight': 'right',
      };

      const direction = keyMap[e.key];
      if (direction) {
        e.preventDefault();
        makeMove(direction);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [gameState]);

  const fetchGameState = async () => {
    try {
      const response = await fetch(`${API_URL}/api/game/state`);
      const data = await response.json();
      setGameState(data);
    } catch (error) {
      console.error('Error fetching game state:', error);
      setMessage('Error loading game');
    }
  };

  const makeMove = async (direction: string) => {
    if (loading) return;
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/game/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ direction }),
      });

      if (response.ok) {
        const data = await response.json();
        setGameState(data);
        
        if (data.won && !gameState?.won) {
          setMessage(`🎉 You reached ${data.target_tile}!`);
        } else if (data.game_over) {
          setMessage('Game Over! No more moves available.');
        }
      } else {
        // Invalid move, just ignore
      }
    } catch (error) {
      console.error('Error making move:', error);
    } finally {
      setLoading(false);
    }
  };

  const newGame = async (boardSize: number = 4, targetTile: number = 2048) => {
    setLoading(true);
    setMessage('');

    try {
      const response = await fetch(`${API_URL}/api/game/new`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board_size: boardSize, target_tile: targetTile }),
      });

      const data = await response.json();
      setGameState(data);
    } catch (error) {
      console.error('Error starting new game:', error);
      setMessage('Error starting new game');
    } finally {
      setLoading(false);
    }
  };

  // Touch/swipe handling
  const [touchStart, setTouchStart] = useState<{ x: number; y: number } | null>(null);

  const handleTouchStart = (e: React.TouchEvent) => {
    const touch = e.touches[0];
    setTouchStart({ x: touch.clientX, y: touch.clientY });
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (!touchStart) return;

    const touch = e.changedTouches[0];
    const deltaX = touch.clientX - touchStart.x;
    const deltaY = touch.clientY - touchStart.y;
    const minSwipeDistance = 50;

    if (Math.abs(deltaX) > Math.abs(deltaY)) {
      // Horizontal swipe
      if (Math.abs(deltaX) > minSwipeDistance) {
        makeMove(deltaX > 0 ? 'right' : 'left');
      }
    } else {
      // Vertical swipe
      if (Math.abs(deltaY) > minSwipeDistance) {
        makeMove(deltaY > 0 ? 'down' : 'up');
      }
    }

    setTouchStart(null);
  };

  if (!gameState) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="text-xl">Loading game...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="max-w-2xl mx-auto px-4">
        <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">2048</h1>
        
        <GameControls
          gameState={gameState}
          onNewGame={newGame}
          message={message}
        />
        
        <div
          onTouchStart={handleTouchStart}
          onTouchEnd={handleTouchEnd}
          className="touch-none"
        >
          <GameBoard board={gameState.board} />
        </div>
        
        <div className="mt-6 text-center text-gray-600">
          <p className="mb-2">Use arrow keys to play</p>
          <p className="text-sm">Combine tiles to reach {gameState.target_tile}!</p>
        </div>
      </div>
    </div>
  );
}