"""2048 Game Logic for Browser Environment"""

import random
import numpy as np
from typing import Tuple, Optional, List


class Game2048:
    """Browser-based 2048 game implementation with configurable target"""

    def __init__(self, size: int = 4, target_tile: int = 2048):
        self.size = size
        self.target_tile = target_tile
        self.board = np.zeros((size, size), dtype=int)
        self.score = 0
        self.game_over = False
        self.moves_made = 0
        self.won = False

        # Start with 2 random tiles
        self.add_random_tile()
        self.add_random_tile()

        # Track initial highest tile for reward calculation
        self.initial_highest_tile = int(self.board.max())

    def add_random_tile(self) -> bool:
        """Add a random 2 or 4 tile to an empty position"""
        empty_cells = [
            (i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0
        ]

        if not empty_cells:
            return False

        i, j = random.choice(empty_cells)
        # 90% chance of 2, 10% chance of 4
        self.board[i, j] = 2 if random.random() < 0.9 else 4
        return True

    def compress(self, row: np.ndarray) -> Tuple[np.ndarray, int]:
        """Compress a row by moving all non-zero elements to the left and merging"""
        new_row = np.zeros_like(row)
        pos = 0
        score = 0

        # Move all non-zero elements to the left
        for num in row:
            if num != 0:
                new_row[pos] = num
                pos += 1

        # Merge adjacent equal elements
        i = 0
        while i < len(new_row) - 1:
            if new_row[i] != 0 and new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                score += new_row[i]
                new_row[i + 1] = 0
                i += 2
            else:
                i += 1

        # Compress again after merging
        final_row = np.zeros_like(row)
        pos = 0
        for num in new_row:
            if num != 0:
                final_row[pos] = num
                pos += 1

        return final_row, score

    def move(self, direction: str) -> bool:
        """Make a move in the specified direction"""
        if self.game_over:
            return False

        direction = direction.lower()
        if direction not in ["up", "down", "left", "right"]:
            return False

        original_board = self.board.copy()
        move_score = 0

        if direction == "left":
            for i in range(self.size):
                self.board[i], row_score = self.compress(self.board[i])
                move_score += row_score

        elif direction == "right":
            for i in range(self.size):
                reversed_row = self.board[i][::-1]
                compressed, row_score = self.compress(reversed_row)
                self.board[i] = compressed[::-1]
                move_score += row_score

        elif direction == "up":
            for j in range(self.size):
                column = self.board[:, j]
                compressed, col_score = self.compress(column)
                self.board[:, j] = compressed
                move_score += col_score

        elif direction == "down":
            for j in range(self.size):
                column = self.board[:, j][::-1]
                compressed, col_score = self.compress(column)
                self.board[:, j] = compressed[::-1]
                move_score += col_score

        # Check if the board changed
        if not np.array_equal(original_board, self.board):
            self.score += move_score
            self.moves_made += 1
            self.add_random_tile()
            self.check_game_status()
            return True

        return False

    def check_game_status(self):
        """Check if the game is won or over"""
        # Check if target tile is reached
        if not self.won and self.board.max() >= self.target_tile:
            self.won = True

        # Check if game is over (no valid moves)
        # Check for empty cells
        if 0 in self.board:
            self.game_over = False
            return

        # Check for possible merges
        for i in range(self.size):
            for j in range(self.size):
                current = self.board[i, j]
                # Check right neighbor
                if j < self.size - 1 and current == self.board[i, j + 1]:
                    self.game_over = False
                    return
                # Check bottom neighbor
                if i < self.size - 1 and current == self.board[i + 1, j]:
                    self.game_over = False
                    return

        self.game_over = True

    def get_state(self) -> dict:
        """Get the current game state as a dictionary"""
        return {
            "board": self.board.tolist(),
            "score": int(self.score),
            "moves": int(self.moves_made),
            "game_over": bool(self.game_over),
            "won": bool(self.won),
            "highest_tile": int(self.board.max()),
            "initial_highest_tile": int(self.initial_highest_tile),
            "target_tile": self.target_tile,
            "board_size": self.size,
        }

    def set_board(self, board: List[List[int]], score: int = 0, moves: int = 0):
        """Set a specific board configuration (for testing)"""
        self.board = np.array(board, dtype=int)
        self.score = score
        self.moves_made = moves
        self.check_game_status()

    def reset(self, size: Optional[int] = None, target_tile: Optional[int] = None):
        """Reset the game to initial state

        Args:
            size: Optional new board size
            target_tile: Optional new target tile
        """
        if size is not None:
            self.size = size
        if target_tile is not None:
            self.target_tile = target_tile

        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.game_over = False
        self.won = False
        self.moves_made = 0
        self.add_random_tile()
        self.add_random_tile()

        # Track initial highest tile after reset
        self.initial_highest_tile = int(self.board.max())

    def can_move(self) -> dict:
        """Check which moves are valid"""
        valid_moves = {"up": False, "down": False, "left": False, "right": False}

        if self.game_over:
            return valid_moves

        # Test each direction without modifying the actual board
        original_board = self.board.copy()

        for direction in ["up", "down", "left", "right"]:
            test_board = original_board.copy()
            self.board = test_board

            # Try the move
            if direction == "left":
                for i in range(self.size):
                    compressed, _ = self.compress(self.board[i])
                    if not np.array_equal(self.board[i], compressed):
                        valid_moves[direction] = True
                        break

            elif direction == "right":
                for i in range(self.size):
                    reversed_row = self.board[i][::-1]
                    compressed, _ = self.compress(reversed_row)
                    if not np.array_equal(reversed_row, compressed):
                        valid_moves[direction] = True
                        break

            elif direction == "up":
                for j in range(self.size):
                    column = self.board[:, j]
                    compressed, _ = self.compress(column)
                    if not np.array_equal(column, compressed):
                        valid_moves[direction] = True
                        break

            elif direction == "down":
                for j in range(self.size):
                    column = self.board[:, j][::-1]
                    compressed, _ = self.compress(column)
                    if not np.array_equal(column, compressed):
                        valid_moves[direction] = True
                        break

        # Restore original board
        self.board = original_board
        return valid_moves
