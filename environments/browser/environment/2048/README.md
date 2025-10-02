# 2048 Game for Browser Environment

A browser-based implementation of the 2048 game with configurable target tiles and reward system for RL evaluation.

## Features

- **Configurable Target Tile**: Set any power of 2 as target (64, 128, 256, 512, 1024, 2048, etc.)
- **Logarithmic Reward Scaling**: Smooth reward progression using `log(highest_tile) / log(target)`
- **Efficiency Tracking**: Monitor score-to-moves ratio
- **Flexible Board Size**: Support for 3x3 to 6x6 grids
- **Full Evaluation API**: Compatible with RL evaluation system

## Architecture

### Backend (FastAPI)
- Core game logic in `game.py`
- RESTful API endpoints for game control
- Evaluation endpoints for RL agents
- SQLite persistence (optional)

### Frontend (Next.js + React)
- Responsive game board with smooth animations
- Keyboard and touch controls
- Real-time score and progress tracking
- Customizable game parameters

## Running the Game

### Standalone
```bash
python launch.py --frontend-port 3001 --backend-port 5001
```

### With Browser Environment
The game integrates with the browser environment's setup and evaluation system.

## API Endpoints

### Core Game
- `POST /api/game/new` - Start new game
- `GET /api/game/state` - Get current state
- `POST /api/game/move` - Make a move
- `POST /api/game/set_target` - Set target tile

### Evaluation
- `GET /api/eval/stats` - Get comprehensive stats
- `GET /api/eval/max_number` - Get highest tile
- `GET /api/eval/efficiency` - Get efficiency ratio
- `POST /api/eval/set_board` - Set specific board
- `POST /api/eval/reset` - Reset game

## Evaluators

- `game_2048_max_number` - Check if target tile reached (logarithmic reward)
- `game_2048_efficiency` - Evaluate score/moves ratio
- `game_2048_score_reached` - Check if target score reached
- `game_2048_game_won` - Check if game is won
- `game_2048_game_over` - Check if game is over
- `game_2048_moves_made` - Check minimum moves made

## Setup Tools

- `game_2048_board` - Initialize game with size and target
- `game_2048_set_board` - Set specific board state
- `game_2048_near_win` - Set board near winning
- `game_2048_navigate` - Navigate to game URL
- `game_2048_reset` - Reset to initial state

## Reward System

The reward system matches the text-2048 environment:

1. **Max Number Reward**: `min(1.0, log(highest_tile) / log(target))`
   - Logarithmic scaling for smooth progression
   - Reaches 1.0 when target tile is achieved

2. **Efficiency Reward**: `min(1.0, ratio / min_ratio)`
   - Linear scaling based on score/moves ratio
   - Encourages efficient gameplay

## Development

### Backend Requirements
- Python 3.8+
- FastAPI
- NumPy
- uvicorn

### Frontend Requirements
- Node.js 16+
- Next.js 14
- React 18
- Tailwind CSS

## Testing

The game can be tested with the browser environment's evaluation system:

```python
# Example evaluation
ctx = Context()
result = await game_2048_max_number(ctx, target=2048)
```