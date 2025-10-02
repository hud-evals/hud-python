"""FastAPI backend for 2048 game"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import sqlite3
import json
from game import Game2048

app = FastAPI(title="2048 Game API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Different port from todo app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global game instance (in production, would use sessions/database)
game = Game2048()


# Pydantic models
class NewGameRequest(BaseModel):
    board_size: int = 4
    target_tile: int = 2048


class MoveRequest(BaseModel):
    direction: str  # up, down, left, right


class SetBoardRequest(BaseModel):
    board: List[List[int]]
    score: Optional[int] = 0
    moves: Optional[int] = 0


class SetTargetRequest(BaseModel):
    target_tile: int


class GameState(BaseModel):
    board: List[List[int]]
    score: int
    moves: int
    game_over: bool
    won: bool
    highest_tile: int
    initial_highest_tile: int
    target_tile: int
    board_size: int


class EvaluationStats(BaseModel):
    board: List[List[int]]
    score: int
    moves: int
    highest_tile: int
    target_tile: int
    efficiency: float
    game_over: bool
    won: bool
    valid_moves: dict


# === CORE GAME API ROUTES ===


@app.get("/api/status")
def status():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/game/new", response_model=GameState)
def new_game(request: NewGameRequest):
    """Start a new game with specified parameters"""
    global game
    game = Game2048(size=request.board_size, target_tile=request.target_tile)
    return game.get_state()


@app.get("/api/game/state", response_model=GameState)
def get_game_state():
    """Get current game state"""
    return game.get_state()


@app.post("/api/game/move", response_model=GameState)
def make_move(request: MoveRequest):
    """Make a move in the specified direction"""
    valid = game.move(request.direction)
    if not valid and not game.game_over:
        raise HTTPException(status_code=400, detail="Invalid move")
    return game.get_state()


@app.post("/api/game/set_target", response_model=GameState)
def set_target(request: SetTargetRequest):
    """Set the target tile for the game"""
    game.target_tile = request.target_tile
    game.check_game_status()  # Re-check win condition
    return game.get_state()


@app.get("/api/game/valid_moves")
def get_valid_moves():
    """Get which moves are currently valid"""
    return game.can_move()


# === EVALUATION API ROUTES ===


@app.get("/api/eval/health")
def eval_health():
    """Health check endpoint for evaluation system"""
    return {
        "status": "healthy",
        "game_active": not game.game_over,
        "highest_tile": int(game.board.max()),
        "target_tile": game.target_tile,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/eval/stats", response_model=EvaluationStats)
def get_evaluation_stats():
    """Comprehensive evaluation statistics for the game"""
    state = game.get_state()
    efficiency = state["score"] / state["moves"] if state["moves"] > 0 else 0.0

    return EvaluationStats(
        board=state["board"],
        score=state["score"],
        moves=state["moves"],
        highest_tile=state["highest_tile"],
        target_tile=state["target_tile"],
        efficiency=efficiency,
        game_over=state["game_over"],
        won=state["won"],
        valid_moves=game.can_move(),
    )


@app.get("/api/eval/max_number")
def get_max_number():
    """Get the highest tile value for evaluation"""
    state = game.get_state()
    return {
        "highest_tile": state["highest_tile"],
        "target_tile": state["target_tile"],
        "progress": state["highest_tile"] / state["target_tile"] if state["target_tile"] > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/eval/efficiency")
def get_efficiency():
    """Get the game efficiency (score/moves ratio)"""
    state = game.get_state()
    efficiency = state["score"] / state["moves"] if state["moves"] > 0 else 0.0

    return {
        "score": state["score"],
        "moves": state["moves"],
        "efficiency": efficiency,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/eval/board")
def get_board():
    """Get current board state for evaluation"""
    state = game.get_state()
    return {
        "board": state["board"],
        "board_size": state["board_size"],
        "empty_cells": sum(1 for row in state["board"] for cell in row if cell == 0),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/eval/set_board", response_model=GameState)
def set_board(request: SetBoardRequest):
    """Set a specific board configuration for testing"""
    try:
        game.set_board(request.board, request.score, request.moves)
        return game.get_state()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/eval/reset", response_model=GameState)
def reset_game():
    """Reset game to initial state"""
    game.reset()
    return game.get_state()


@app.post("/api/eval/seed")
def seed_test_board():
    """Seed the board with a test configuration"""
    # Create a board that's close to winning
    test_board = [[1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0], [0, 0, 0, 0]]
    game.set_board(test_board, score=10000, moves=100)

    return {
        "message": "Test board seeded successfully",
        "highest_tile": 1024,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/eval/seed_custom")
def seed_custom_board(board: List[List[int]]):
    """Seed the board with a custom configuration"""
    try:
        game.set_board(board)
        state = game.get_state()
        return {
            "message": "Custom board seeded successfully",
            "highest_tile": state["highest_tile"],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/eval/can_move")
def can_move():
    """Check if any moves are available"""
    valid_moves = game.can_move()
    has_moves = any(valid_moves.values())

    return {
        "can_move": has_moves,
        "valid_moves": valid_moves,
        "game_over": game.game_over,
        "timestamp": datetime.now().isoformat(),
    }
