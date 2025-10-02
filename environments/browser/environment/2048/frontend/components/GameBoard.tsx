import React from 'react';
import GameTile from './GameTile';

interface GameBoardProps {
  board: number[][];
}

export default function GameBoard({ board }: GameBoardProps) {
  const boardSize = board.length;

  return (
    <div className="relative bg-gray-300 rounded-lg p-2 shadow-lg">
      <div 
        className="grid gap-2"
        style={{
          gridTemplateColumns: `repeat(${boardSize}, 1fr)`,
        }}
      >
        {board.map((row, i) =>
          row.map((value, j) => (
            <GameTile
              key={`${i}-${j}`}
              value={value}
              position={{ row: i, col: j }}
            />
          ))
        )}
      </div>
    </div>
  );
}