# Todo App

Simple todo list application with Next.js frontend and FastAPI backend, fully integrated with the HUD evaluation system.

## Tech Stack

- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **Backend**: FastAPI, SQLite, uv for dependency management
- **Evaluation**: Comprehensive API endpoints for testing

## Development

```bash
# Backend
cd backend && uv run uvicorn main:app --reload

# Frontend  
cd frontend && npm install && npm run dev
```

## Launching

```python
await client.call_tool("launch_app", {"app_name": "todo"})
```

## Evaluation Integration

### Backend API Endpoints
- `GET /api/eval/health` - Health check
- `GET /api/eval/stats` - Comprehensive statistics
- `GET /api/eval/has_todo?text=` - Check if todo exists
- `GET /api/eval/completion_rate` - Completion percentage
- `POST /api/eval/seed` - Seed test data
- `DELETE /api/eval/reset` - Reset database

### Controller Components
- **Evaluators**: `TodoCompletedEvaluator`, `TodoExistsEvaluator`, `CompositeEvaluator`
- **Setup Tools**: `TodoSeedSetup`, `TodoResetSetup`, `TodoCustomSeedSetup`
- **Problems**: `TodoBasicUsageProblem`, `TodoCompositeWeightedProblem`

### Usage Examples

```python
# Complete problem execution
await setup({"name": "todo_basic_usage"})
await evaluate({"name": "todo_basic_usage"})

# Direct function calls
await setup({"name": "todo_reset", "arguments": {}})
await evaluate({"name": "todo_completion_rate", "arguments": {"min_rate": 0.5}})

# MCP resource discovery
todo_evaluators = await client.read_resource("evaluators://todo")
```

## Database Schema

```sql
CREATE TABLE items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Testing

### Manual
1. Launch app: `await launch_app("todo")`
2. Access at http://localhost:3000
3. Run evaluations

### Automated
```bash
# Test APIs
curl http://localhost:5000/api/eval/health
curl http://localhost:5000/api/eval/stats

# Test MCP tools
await setup({"name": "todo_basic_usage"})
await evaluate({"name": "todo_basic_usage"})
``` 