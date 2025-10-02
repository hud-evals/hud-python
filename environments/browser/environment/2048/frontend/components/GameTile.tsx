import React from 'react';

interface GameTileProps {
  value: number;
  position: { row: number; col: number };
}

export default function GameTile({ value }: GameTileProps) {
  const getTileColor = (val: number): string => {
    const colors: { [key: number]: string } = {
      0: 'bg-gray-200',
      2: 'bg-yellow-100',
      4: 'bg-yellow-200',
      8: 'bg-orange-300',
      16: 'bg-orange-400',
      32: 'bg-orange-500',
      64: 'bg-red-400',
      128: 'bg-yellow-300',
      256: 'bg-yellow-400',
      512: 'bg-yellow-500',
      1024: 'bg-yellow-600',
      2048: 'bg-yellow-700',
      4096: 'bg-purple-600',
      8192: 'bg-purple-700',
    };
    return colors[val] || 'bg-purple-800';
  };

  const getTextSize = (val: number): string => {
    if (val === 0) return '';
    if (val < 100) return 'text-3xl';
    if (val < 1000) return 'text-2xl';
    return 'text-xl';
  };

  const getTextColor = (val: number): string => {
    return val > 4 ? 'text-white' : 'text-gray-800';
  };

  return (
    <div
      className={`
        aspect-square rounded flex items-center justify-center font-bold
        transition-all duration-150 ease-in-out
        ${getTileColor(value)}
        ${getTextColor(value)}
        ${getTextSize(value)}
      `}
    >
      {value > 0 && value}
    </div>
  );
}