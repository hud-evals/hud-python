"""
FastAPI backend for MCP Multi-Tools environment.
Handles API integrations and state management.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Multi-Tools Environment", version="0.1.0")


# ===========================================
# Scratchpad State (In-Memory)
# ===========================================
class ScratchpadState:
    def __init__(self):
        self.data: Dict[str, str] = {}

    def write(self, key: str, value: str) -> None:
        self.data[key] = value

    def read(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def list_keys(self) -> List[str]:
        return list(self.data.keys())

    def delete(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
            return True
        return False

    def clear(self) -> None:
        self.data.clear()


scratchpad = ScratchpadState()


# ===========================================
# Document Store (File-based)
# ===========================================
class DocumentStore:
    """Manages a collection of text documents for retrieval tasks."""
    
    def __init__(self, documents_dir: str = "/app/documents"):
        self.documents_dir = documents_dir
        self._cache: Dict[str, str] = {}  # Cache loaded documents
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all available documents with metadata."""
        docs = []
        if not os.path.exists(self.documents_dir):
            return docs
        
        for filename in os.listdir(self.documents_dir):
            if filename.endswith(('.txt', '.md')):
                filepath = os.path.join(self.documents_dir, filename)
                doc_id = filename.rsplit('.', 1)[0]
                content = self._load_document(doc_id)
                
                # Calculate approximate page count (assuming ~3000 chars per page)
                char_count = len(content) if content else 0
                page_count = max(1, char_count // 3000)
                
                docs.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "size_chars": char_count,
                    "approx_pages": page_count,
                })
        return docs
    
    def _load_document(self, doc_id: str) -> Optional[str]:
        """Load document content from file."""
        if doc_id in self._cache:
            return self._cache[doc_id]
        
        for ext in ['.txt', '.md']:
            filepath = os.path.join(self.documents_dir, f"{doc_id}{ext}")
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._cache[doc_id] = content
                return content
        return None
    
    def read_document(self, doc_id: str, start_page: int = 1, end_page: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Read document content, optionally paginated."""
        content = self._load_document(doc_id)
        if content is None:
            return None
        
        # Approximate pagination (3000 chars per page)
        chars_per_page = 3000
        total_pages = max(1, len(content) // chars_per_page + (1 if len(content) % chars_per_page else 0))
        
        start_idx = (start_page - 1) * chars_per_page
        if end_page:
            end_idx = min(end_page * chars_per_page, len(content))
        else:
            end_idx = min(start_page * chars_per_page, len(content))
        
        return {
            "doc_id": doc_id,
            "content": content[start_idx:end_idx],
            "page": start_page,
            "end_page": end_page or start_page,
            "total_pages": total_pages,
            "total_chars": len(content),
        }
    
    def search_document(self, doc_id: str, query: str, context_chars: int = 200) -> List[Dict[str, Any]]:
        """Search for query in document, return matching sections with context."""
        content = self._load_document(doc_id)
        if content is None:
            return []
        
        results = []
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Find all occurrences
        start = 0
        while True:
            idx = content_lower.find(query_lower, start)
            if idx == -1:
                break
            
            # Extract context around match
            context_start = max(0, idx - context_chars)
            context_end = min(len(content), idx + len(query) + context_chars)
            
            # Find line number
            line_num = content[:idx].count('\n') + 1
            
            results.append({
                "match_position": idx,
                "line_number": line_num,
                "context": content[context_start:context_end],
                "match_text": content[idx:idx + len(query)],
            })
            
            start = idx + 1
            
            # Limit to 20 results
            if len(results) >= 20:
                break
        
        return results


document_store = DocumentStore()


# ===========================================
# Request Models
# ===========================================
class ScratchpadWriteRequest(BaseModel):
    key: str
    value: str


class ScratchpadReadRequest(BaseModel):
    key: str


class ScratchpadDeleteRequest(BaseModel):
    key: str


class SetupRequest(BaseModel):
    initial_data: Optional[Dict[str, str]] = None  # Pre-populate scratchpad


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class FetchRequest(BaseModel):
    url: str


class SupabaseQueryRequest(BaseModel):
    table: str
    select: str = "*"
    filters: Optional[Dict[str, Any]] = None  # eq filters: {col: value}
    limit: Optional[int] = None  # no default - agent must specify or get all


class SupabaseInsertRequest(BaseModel):
    table: str
    data: Dict[str, Any]


class SupabaseUpdateRequest(BaseModel):
    table: str
    data: Dict[str, Any]
    filters: Dict[str, Any]


class LinearListIssuesRequest(BaseModel):
    team_id: Optional[str] = None
    limit: int = 10


class LinearCreateIssueRequest(BaseModel):
    title: str
    description: str
    team_id: str


class LinearUpdateIssueRequest(BaseModel):
    issue_id: str
    status: Optional[str] = None
    title: Optional[str] = None


class EvaluateRequest(BaseModel):
    expected_keys: Optional[List[str]] = None
    expected_values: Optional[Dict[str, str]] = None
    exact_values: Optional[Dict[str, str]] = None
    min_value_length: int = 0


# GitHub API Request Models
class GitHubReadFileRequest(BaseModel):
    owner: str
    repo: str
    path: str
    ref: Optional[str] = None  # branch, tag, or commit SHA


class GitHubListFilesRequest(BaseModel):
    owner: str
    repo: str
    path: str = ""
    ref: Optional[str] = None


class GitHubSearchCodeRequest(BaseModel):
    owner: str
    repo: str
    query: str
    path: Optional[str] = None  # Limit to specific path


class DocumentReadRequest(BaseModel):
    doc_id: str
    start_page: int = 1
    end_page: Optional[int] = None


class DocumentSearchRequest(BaseModel):
    doc_id: str
    query: str
    context_chars: int = 200


class ExecutePythonRequest(BaseModel):
    code: str
    timeout: int = 30  # seconds


# ===========================================
# Health & Setup Endpoints
# ===========================================
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy"}


@app.post("/setup")
async def setup(req: Optional[SetupRequest] = None) -> Dict[str, Any]:
    scratchpad.clear()
    
    # Pre-populate scratchpad with initial data if provided
    if req and req.initial_data:
        for key, value in req.initial_data.items():
            scratchpad.write(key, value)
        return {"ok": True, "message": f"Environment reset. Pre-loaded {len(req.initial_data)} keys."}
    
    return {"ok": True, "message": "Environment reset"}


# ===========================================
# Scratchpad Endpoints
# ===========================================
@app.post("/scratchpad/write")
async def scratchpad_write(req: ScratchpadWriteRequest) -> Dict[str, Any]:
    scratchpad.write(req.key, req.value)
    return {"ok": True, "key": req.key}


@app.post("/scratchpad/read")
async def scratchpad_read(req: ScratchpadReadRequest) -> Dict[str, Any]:
    value = scratchpad.read(req.key)
    if value is None:
        return {"ok": False, "error": f"Key '{req.key}' not found"}
    return {"ok": True, "key": req.key, "value": value}


@app.get("/scratchpad/list")
async def scratchpad_list() -> Dict[str, Any]:
    return {"ok": True, "keys": scratchpad.list_keys()}


@app.post("/scratchpad/delete")
async def scratchpad_delete(req: ScratchpadDeleteRequest) -> Dict[str, Any]:
    deleted = scratchpad.delete(req.key)
    return {"ok": deleted, "key": req.key}


# ===========================================
# Exa Search Endpoints
# ===========================================
@app.post("/exa/search")
async def exa_search(req: SearchRequest) -> List[Dict[str, str]]:
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="EXA_API_KEY not set")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "query": req.query,
                "numResults": req.max_results,
                "type": "keyword",
                "contents": {"text": {"maxCharacters": 1000}},
            },
        )
        response.raise_for_status()
        data = response.json()

    results = []
    for item in data.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "text": item.get("text", "")[:500] if item.get("text") else "",
        })
    return results


@app.post("/exa/fetch")
async def exa_fetch(req: FetchRequest) -> Dict[str, str]:
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="EXA_API_KEY not set")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.exa.ai/contents",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "urls": [req.url],
                "text": {"maxCharacters": 5000},
            },
        )
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])
    if results:
        return {"content": results[0].get("text", "No content available")}
    return {"content": "No content available"}


# ===========================================
# Document Store Endpoints
# ===========================================
@app.get("/documents/list")
async def list_documents() -> Dict[str, Any]:
    """List all available documents."""
    docs = document_store.list_documents()
    return {"ok": True, "documents": docs, "count": len(docs)}


@app.post("/documents/read")
async def read_document(req: DocumentReadRequest) -> Dict[str, Any]:
    """Read document content with optional pagination."""
    result = document_store.read_document(req.doc_id, req.start_page, req.end_page)
    if result is None:
        return {"ok": False, "error": f"Document '{req.doc_id}' not found"}
    return {"ok": True, **result}


@app.post("/documents/search")
async def search_document(req: DocumentSearchRequest) -> Dict[str, Any]:
    """Search within a document for specific content."""
    # First check if document exists
    content = document_store._load_document(req.doc_id)
    if content is None:
        return {"ok": False, "error": f"Document '{req.doc_id}' not found", "matches": []}
    
    matches = document_store.search_document(req.doc_id, req.query, req.context_chars)
    return {"ok": True, "doc_id": req.doc_id, "query": req.query, "matches": matches, "match_count": len(matches)}


# ===========================================
# Python Code Execution Endpoint
# ===========================================
import subprocess
import tempfile


@app.post("/execute_python")
async def execute_python(req: ExecutePythonRequest) -> Dict[str, Any]:
    """Execute Python code in a sandboxed environment."""
    
    # Create a wrapper script that includes environment setup
    wrapper_code = '''
import os
import sys
import json

# Set up Supabase credentials from environment
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY", "")

# User code starts here
''' + req.code
    
    try:
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper_code)
            temp_path = f.name
        
        # Execute with timeout
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            text=True,
            timeout=req.timeout,
            env={
                **os.environ,
                'PYTHONPATH': '/app',
            }
        )
        
        # Clean up
        os.unlink(temp_path)
        
        return {
            "ok": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"Code execution timed out after {req.timeout} seconds",
            "stdout": "",
            "stderr": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "stdout": "",
            "stderr": "",
        }


# ===========================================
# GitHub API Endpoints
# ===========================================
GITHUB_API_URL = "https://api.github.com"


def _get_github_headers() -> Dict[str, str]:
    """Get GitHub API headers. Token is optional for public repos."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "MCP-MultiTools/1.0",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@app.post("/github/read_file")
async def github_read_file(req: GitHubReadFileRequest) -> Dict[str, Any]:
    """Read a file from a GitHub repository."""
    headers = _get_github_headers()
    
    # Build URL
    url = f"{GITHUB_API_URL}/repos/{req.owner}/{req.repo}/contents/{req.path}"
    if req.ref:
        url += f"?ref={req.ref}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code == 404:
                return {"ok": False, "error": f"File not found: {req.owner}/{req.repo}/{req.path}"}
            if response.status_code == 403:
                return {"ok": False, "error": "Rate limited or access denied. Try adding GITHUB_TOKEN."}
            if response.status_code >= 400:
                return {"ok": False, "error": f"GitHub API error: {response.text}"}
            
            data = response.json()
            
            # Check if it's a file (not a directory)
            if isinstance(data, list):
                return {"ok": False, "error": f"Path is a directory, not a file: {req.path}"}
            
            # Decode content (base64 encoded)
            import base64
            content = base64.b64decode(data.get("content", "")).decode("utf-8")
            
            return {
                "ok": True,
                "owner": req.owner,
                "repo": req.repo,
                "path": req.path,
                "ref": req.ref or data.get("sha", "main"),
                "content": content,
                "size": data.get("size", len(content)),
                "sha": data.get("sha"),
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/github/list_files")
async def github_list_files(req: GitHubListFilesRequest) -> Dict[str, Any]:
    """List files in a GitHub repository directory."""
    headers = _get_github_headers()
    
    # Build URL
    url = f"{GITHUB_API_URL}/repos/{req.owner}/{req.repo}/contents/{req.path}"
    if req.ref:
        url += f"?ref={req.ref}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code == 404:
                return {"ok": False, "error": f"Path not found: {req.owner}/{req.repo}/{req.path}"}
            if response.status_code >= 400:
                return {"ok": False, "error": f"GitHub API error: {response.text}"}
            
            data = response.json()
            
            # If it's a single file, return error
            if not isinstance(data, list):
                return {"ok": False, "error": f"Path is a file, not a directory: {req.path}"}
            
            files = []
            for item in data:
                files.append({
                    "name": item.get("name"),
                    "path": item.get("path"),
                    "type": item.get("type"),  # "file" or "dir"
                    "size": item.get("size", 0),
                })
            
            return {
                "ok": True,
                "owner": req.owner,
                "repo": req.repo,
                "path": req.path,
                "files": files,
                "count": len(files),
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/github/search_code")
async def github_search_code(req: GitHubSearchCodeRequest) -> Dict[str, Any]:
    """Search for code in a GitHub repository."""
    headers = _get_github_headers()
    
    # Build search query
    query = f"{req.query} repo:{req.owner}/{req.repo}"
    if req.path:
        query += f" path:{req.path}"
    
    url = f"{GITHUB_API_URL}/search/code?q={query}&per_page=10"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code == 403:
                return {"ok": False, "error": "Rate limited. GitHub search requires authentication. Add GITHUB_TOKEN."}
            if response.status_code >= 400:
                return {"ok": False, "error": f"GitHub API error: {response.text}"}
            
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append({
                    "path": item.get("path"),
                    "name": item.get("name"),
                    "sha": item.get("sha"),
                    "url": item.get("html_url"),
                    "score": item.get("score"),
                })
            
            return {
                "ok": True,
                "query": req.query,
                "repo": f"{req.owner}/{req.repo}",
                "results": results,
                "total_count": data.get("total_count", 0),
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ===========================================
# Supabase Endpoints
# ===========================================
# def _get_supabase_client() -> tuple[str, str]:
#     url = os.getenv("SUPABASE_URL")
#     key = os.getenv("SUPABASE_API_KEY")
#     if not url or not key:
#         raise HTTPException(status_code=400, detail="SUPABASE_URL or SUPABASE_API_KEY not set")
#     return url, key


# @app.post("/supabase/query")
# async def supabase_query(req: SupabaseQueryRequest) -> Dict[str, Any]:
#     url, key = _get_supabase_client()
#     query_url = f"{url}/rest/v1/{req.table}?select={req.select}"
    
#     if req.limit is not None:
#         query_url += f"&limit={req.limit}"
    
#     if req.filters:
#         for col, val in req.filters.items():
#             if val is None:
#                 query_url += f"&{col}=is.null"
#             else:
#                 query_url += f"&{col}=eq.{val}"

#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             response = await client.get(
#                 query_url,
#                 headers={
#                     "apikey": key,
#                     "Authorization": f"Bearer {key}",
#                     "Content-Type": "application/json",
#                 },
#             )
#             if response.status_code == 404:
#                 return {"ok": False, "error": f"Table '{req.table}' not found", "data": []}
#             if response.status_code >= 400:
#                 return {"ok": False, "error": f"Query failed: {response.text}", "data": []}
#             return {"ok": True, "data": response.json()}
#     except Exception as e:
#         return {"ok": False, "error": str(e), "data": []}


# @app.post("/supabase/insert")
# async def supabase_insert(req: SupabaseInsertRequest) -> Dict[str, Any]:
#     url, key = _get_supabase_client()

#     async with httpx.AsyncClient(timeout=30.0) as client:
#         response = await client.post(
#             f"{url}/rest/v1/{req.table}",
#             headers={
#                 "apikey": key,
#                 "Authorization": f"Bearer {key}",
#                 "Content-Type": "application/json",
#                 "Prefer": "return=representation",
#             },
#             json=req.data,
#         )
#         response.raise_for_status()
#         return {"ok": True, "data": response.json()}


# @app.post("/supabase/update")
# async def supabase_update(req: SupabaseUpdateRequest) -> Dict[str, Any]:
#     url, key = _get_supabase_client()
#     query_url = f"{url}/rest/v1/{req.table}"
    
#     filter_parts = []
#     for col, val in req.filters.items():
#         filter_parts.append(f"{col}=eq.{val}")
#     if filter_parts:
#         query_url += "?" + "&".join(filter_parts)

#     async with httpx.AsyncClient(timeout=30.0) as client:
#         response = await client.patch(
#             query_url,
#             headers={
#                 "apikey": key,
#                 "Authorization": f"Bearer {key}",
#                 "Content-Type": "application/json",
#                 "Prefer": "return=representation",
#             },
#             json=req.data,
#         )
#         response.raise_for_status()
#         return {"ok": True, "data": response.json()}


# ===========================================
# Linear Endpoints
# ===========================================
LINEAR_API_URL = "https://api.linear.app/graphql"


def _get_linear_headers() -> Dict[str, str]:
    api_key = os.getenv("LINEAR_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="LINEAR_API_KEY not set")
    return {"Authorization": api_key, "Content-Type": "application/json"}


@app.post("/linear/list_issues")
async def linear_list_issues(req: LinearListIssuesRequest) -> Dict[str, Any]:
    headers = _get_linear_headers()
    
    # Use different queries based on whether team_id is provided
    if req.team_id:
        query = """
        query($first: Int, $teamId: String!) {
            issues(first: $first, filter: { team: { id: { eq: $teamId } } }) {
                nodes { id title description state { name } priority createdAt }
            }
        }
        """
        variables = {"first": req.limit, "teamId": req.team_id}
    else:
        # Simple query without team filter
        query = """
        query($first: Int) {
            issues(first: $first) {
                nodes { id title description state { name } priority createdAt }
            }
        }
        """
        variables = {"first": req.limit}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(LINEAR_API_URL, headers=headers, json={"query": query, "variables": variables})
            data = response.json()
            
        if "errors" in data:
            return {"ok": False, "error": str(data["errors"]), "issues": []}
        
        issues = data.get("data", {}).get("issues", {}).get("nodes", [])
        return {"ok": True, "issues": issues, "count": len(issues)}
    except Exception as e:
        return {"ok": False, "error": str(e), "issues": []}


@app.post("/linear/create_issue")
async def linear_create_issue(req: LinearCreateIssueRequest) -> Dict[str, Any]:
    headers = _get_linear_headers()
    mutation = """
    mutation($title: String!, $description: String!, $teamId: String!) {
        issueCreate(input: { title: $title, description: $description, teamId: $teamId }) {
            success
            issue { id title url }
        }
    }
    """
    variables = {"title": req.title, "description": req.description, "teamId": req.team_id}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(LINEAR_API_URL, headers=headers, json={"query": mutation, "variables": variables})
        response.raise_for_status()
        data = response.json()
        
    if "errors" in data:
        raise HTTPException(status_code=400, detail=str(data["errors"]))
    
    result = data.get("data", {}).get("issueCreate", {})
    return {"ok": result.get("success", False), "issue": result.get("issue")}


@app.post("/linear/update_issue")
async def linear_update_issue(req: LinearUpdateIssueRequest) -> Dict[str, Any]:
    headers = _get_linear_headers()
    mutation = """
    mutation($issueId: String!, $title: String, $stateId: String) {
        issueUpdate(id: $issueId, input: { title: $title, stateId: $stateId }) {
            success
            issue { id title state { name } }
        }
    }
    """
    variables = {"issueId": req.issue_id}
    if req.title:
        variables["title"] = req.title
    if req.status:
        variables["stateId"] = req.status

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(LINEAR_API_URL, headers=headers, json={"query": mutation, "variables": variables})
        response.raise_for_status()
        data = response.json()
        
    if "errors" in data:
        raise HTTPException(status_code=400, detail=str(data["errors"]))
    
    result = data.get("data", {}).get("issueUpdate", {})
    return {"ok": result.get("success", False), "issue": result.get("issue")}


# ===========================================
# Evaluation Endpoint
# ===========================================
@app.post("/evaluate")
async def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    """Simple evaluation based on scratchpad state."""
    checks_passed = 0
    total_checks = 0
    
    # Check expected keys exist
    if req.expected_keys:
        for key in req.expected_keys:
            total_checks += 1
            value = scratchpad.read(key)
            if value and len(value) >= req.min_value_length:
                checks_passed += 1
    
    # Check expected values (substring match)
    if req.expected_values:
        for key, expected in req.expected_values.items():
            total_checks += 1
            value = scratchpad.read(key)
            if value and expected.lower() in value.lower():
                checks_passed += 1
    
    # Check exact values (strict match)
    if req.exact_values:
        for key, expected in req.exact_values.items():
            total_checks += 1
            value = scratchpad.read(key)
            if value and value.strip() == expected:
                checks_passed += 1
    
    # Calculate reward
    if total_checks == 0:
        reward = 1.0 if scratchpad.list_keys() else 0.0
    else:
        reward = checks_passed / total_checks
    
    return {"reward": reward, "done": True, "content": f"Passed {checks_passed}/{total_checks} checks"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
