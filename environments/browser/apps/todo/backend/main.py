from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import sqlite3
import json

app = FastAPI(title="Todo API with Evaluation", version="0.2.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class Item(BaseModel):
    id: Optional[int] = None
    title: str
    description: str
    completed: bool = False
    created_at: Optional[datetime] = None


class ItemCreate(BaseModel):
    title: str
    description: str
    completed: bool = False


class BulkUpdateRequest(BaseModel):
    item_ids: List[int]
    completed: Optional[bool] = None


class EvaluationStats(BaseModel):
    total_items: int
    completed_items: int
    pending_items: int
    completion_rate: float
    items: List[Item]
    timestamps: dict


# Database setup
def init_db():
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            completed BOOLEAN NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()


# === CORE TODO API ROUTES ===


@app.get("/api/status")
def status():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/items", response_model=List[Item])
def get_items():
    conn = sqlite3.connect("app.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM items ORDER BY created_at DESC")
    items = [dict(row) for row in c.fetchall()]
    conn.close()
    return items


@app.post("/api/items", response_model=Item)
def create_item(item: ItemCreate):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO items (title, description, completed) VALUES (?, ?, ?)",
        (item.title, item.description, item.completed),
    )
    item_id = c.lastrowid
    conn.commit()
    conn.close()

    return get_item(item_id)


@app.get("/api/items/{item_id}", response_model=Item)
def get_item(item_id: int):
    conn = sqlite3.connect("app.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM items WHERE id = ?", (item_id,))
    item = c.fetchone()
    conn.close()

    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    return dict(item)


@app.put("/api/items/{item_id}", response_model=Item)
def update_item(item_id: int, item: ItemCreate):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute(
        "UPDATE items SET title = ?, description = ?, completed = ? WHERE id = ?",
        (item.title, item.description, item.completed, item_id),
    )
    conn.commit()

    if c.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Item not found")

    conn.close()
    return get_item(item_id)


@app.delete("/api/items/{item_id}")
def delete_item(item_id: int):
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("DELETE FROM items WHERE id = ?", (item_id,))
    conn.commit()

    if c.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Item not found")

    conn.close()
    return {"message": "Item deleted successfully"}


# === EVALUATION API ROUTES ===


@app.get("/api/eval/health")
def eval_health():
    """Health check endpoint for evaluation system."""
    try:
        conn = sqlite3.connect("app.db")
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM items")
        count = c.fetchone()[0]
        conn.close()

        return {
            "status": "healthy",
            "database_accessible": True,
            "total_items": count,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}


@app.get("/api/eval/stats", response_model=EvaluationStats)
def get_evaluation_stats():
    """Comprehensive evaluation statistics for the todo app."""
    conn = sqlite3.connect("app.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get total counts
    c.execute("SELECT COUNT(*) as total FROM items")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) as completed FROM items WHERE completed = 1")
    completed = c.fetchone()[0]

    # Get all items with details
    c.execute("SELECT * FROM items ORDER BY created_at DESC")
    items = [dict(row) for row in c.fetchall()]

    # Get timing information
    c.execute("""
        SELECT created_at 
        FROM items 
        ORDER BY created_at DESC 
        LIMIT 1
    """)
    last_created_row = c.fetchone()
    last_created = last_created_row[0] if last_created_row else None

    c.execute("""
        SELECT created_at 
        FROM items 
        WHERE completed = 1 
        ORDER BY created_at DESC 
        LIMIT 1
    """)
    last_completed_row = c.fetchone()
    last_completed = last_completed_row[0] if last_completed_row else None

    conn.close()

    return EvaluationStats(
        total_items=total,
        completed_items=completed,
        pending_items=total - completed,
        completion_rate=completed / total if total > 0 else 0.0,
        items=items,
        timestamps={"last_created": last_created, "last_completed": last_completed},
    )


@app.get("/api/eval/todos", response_model=List[Item])
def get_todos_for_evaluation():
    """Get all todos for evaluation purposes (alias for /api/items)."""
    return get_items()


@app.get("/api/eval/has_todo")
def check_todo_exists(text: str):
    """Check if a todo item exists with specific text in title or description."""
    conn = sqlite3.connect("app.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        """
        SELECT * FROM items 
        WHERE title LIKE ? OR description LIKE ?
        ORDER BY created_at DESC
    """,
        (f"%{text}%", f"%{text}%"),
    )

    items = [dict(row) for row in c.fetchall()]
    conn.close()

    return {
        "exists": len(items) > 0,
        "count": len(items),
        "search_text": text,
        "matches": items,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/eval/bulk_update")
def bulk_update_items(request: BulkUpdateRequest):
    """Update multiple items at once for evaluation purposes."""
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    updated_count = 0
    if request.completed is not None:
        for item_id in request.item_ids:
            c.execute("UPDATE items SET completed = ? WHERE id = ?", (request.completed, item_id))
            if c.rowcount > 0:
                updated_count += 1

    conn.commit()
    conn.close()

    return {
        "message": f"Updated {updated_count} items",
        "updated_count": updated_count,
        "requested_ids": request.item_ids,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/eval/completion_rate")
def get_completion_rate():
    """Get the current completion rate as a percentage."""
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    c.execute("SELECT COUNT(*) as total FROM items")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) as completed FROM items WHERE completed = 1")
    completed = c.fetchone()[0]

    conn.close()

    rate = completed / total if total > 0 else 0.0

    return {
        "completion_rate": rate,
        "completion_percentage": rate * 100,
        "completed_items": completed,
        "total_items": total,
        "timestamp": datetime.now().isoformat(),
    }


# === EVALUATION UTILITY ROUTES ===


@app.post("/api/eval/seed")
def seed_test_data():
    """Seed the database with test data for evaluation purposes."""
    test_items = [
        {"title": "Buy groceries", "description": "Get milk, eggs, and bread", "completed": True},
        {
            "title": "Walk the dog",
            "description": "Take Max for a 30-minute walk",
            "completed": True,
        },
        {
            "title": "Finish project",
            "description": "Complete the Q4 presentation",
            "completed": False,
        },
        {"title": "Call mom", "description": "Weekly check-in call", "completed": False},
        {
            "title": "Schedule dentist",
            "description": "Book appointment for cleaning",
            "completed": False,
        },
    ]

    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    for item in test_items:
        c.execute(
            """
            INSERT INTO items (title, description, completed) 
            VALUES (?, ?, ?)
        """,
            (item["title"], item["description"], item["completed"]),
        )

    conn.commit()
    conn.close()

    return {
        "message": "Test data seeded successfully",
        "items_added": len(test_items),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/eval/seed_custom")
def seed_custom_data(items: List[ItemCreate]):
    """Seed the database with custom test data for evaluation purposes."""
    conn = sqlite3.connect("app.db")
    c = conn.cursor()

    items_added = 0
    for item in items:
        c.execute(
            """
            INSERT INTO items (title, description, completed) 
            VALUES (?, ?, ?)
        """,
            (item.title, item.description if hasattr(item, "description") else "", item.completed),
        )
        items_added += 1

    conn.commit()
    conn.close()

    return {
        "message": "Custom test data seeded successfully",
        "items_added": items_added,
        "timestamp": datetime.now().isoformat(),
    }


@app.delete("/api/eval/reset")
def reset_database():
    """Reset the database to empty state for clean evaluation."""
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("DELETE FROM items")
    conn.commit()
    conn.close()

    return {"message": "Database reset successfully", "timestamp": datetime.now().isoformat()}
