# MCP Multi-Tools Environment: Lessons Learned

## 1. MCP Tool Configuration

Tasks can use **multiple MCP servers** simultaneously. When configured, tools are prefixed with server names (e.g., `local_scratchpad_write`, `supabase_execute_sql`). The `allowed_tools` in agent config must use these prefixed names. Official MCPs (like Supabase) require Bearer token auth in headers.

## 2. Task Structure

```json
{
  "id": "task-example",
  "prompt": "Task description first. Then hints. Then available tools/tables. Finally: Store answer with local_scratchpad_write key 'answer'. Say 'Task completed.'",
  "mcp_config": {
    "local": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "--env-file", "/path/.env", "image:latest"]
    },
    "supabase": {
      "url": "https://mcp.supabase.com/mcp?read_only=true",
      "headers": {"Authorization": "Bearer YOUR_TOKEN"}
    }
  },
  "setup_tool": {"name": "local_setup", "arguments": {}},
  "evaluate_tool": {
    "name": "local_evaluate",
    "arguments": {
      "exact_values": {"answer": "expected_value"}
    }
  },
  "agent_config": {
    "allowed_tools": ["local_scratchpad_write", "supabase_execute_sql"]
  }
}
```

**Key points:**
- `setup_tool` and `evaluate_tool` need the server prefix (`local_`)
- `exact_values` does strict string matching
- Prompt should guide agent to use specific scratchpad key

## 3. Proper Grader Verification

**CRITICAL: Supabase REST API has a 1000 row default limit!**

❌ Wrong way (truncated data):
```python
resp = httpx.get(f'{url}/rest/v1/order_items?select=*', headers=headers)
items = resp.json()  # Only returns first 1000 rows!
```

✅ Correct way (use MCP SQL endpoint):
```python
# Initialize MCP session, then:
result = run_sql("SELECT COUNT(*), SUM(value) FROM table;")
```

Always verify grader answers using `supabase_execute_sql` through the MCP, not the REST API, for tables with >1000 rows.

## 4. Case Sensitivity in Queries

Database values are case-sensitive. If `response_status` contains `'accepted'` (lowercase), querying for `'ACCEPTED'` returns zero rows. Agents should explore data first with `SELECT DISTINCT column FROM table` before writing complex queries.

## 5. Handling Cancelled Tasks in Evaluations

When tasks timeout or get rate-limited, the framework stores `CancelledError` objects instead of traces. The statistics calculation crashes on `.reward` access. Fix: filter out non-trace objects before processing:

```python
task_traces = [t for t in traces if hasattr(t, 'reward')]
```

Run with `--max-concurrent 1` to avoid rate limits from official MCPs.

