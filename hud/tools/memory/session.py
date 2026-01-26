"""Session-based memory tool with optional Qdrant backend.

This tool provides in-session memory storage with add/search operations.
Memory is lost when the session ends unless using a persistent backend.

Backends:
- InMemoryStore: Simple token-overlap similarity (default)
- QdrantBackend: Vector DB with semantic search (requires qdrant-client)
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from mcp.types import TextContent

from hud.tools.memory.base import BaseSessionMemoryTool, MemoryEntry

if TYPE_CHECKING:
    from mcp.types import ContentBlock

    from hud.tools.native_types import NativeToolSpecs

LOGGER = logging.getLogger(__name__)


class SessionMemoryTool(BaseSessionMemoryTool):
    """Add and search short-term memory for a session.

    If Qdrant is available and configured, a remote collection is used.
    Otherwise, an in-memory fallback with token-based similarity is used.

    Parameters:
        action: "add" to store, "search" to retrieve
        text: Content to store or query
        metadata: Optional metadata for stored entries
        top_k: Number of results for search (default: 5)

    Example:
        >>> tool = SessionMemoryTool()
        >>> await tool(action="add", text="User prefers dark mode")
        >>> await tool(action="search", text="user preferences")
    """

    native_specs: ClassVar[NativeToolSpecs] = {}  # Function calling only

    _backend: Any

    def __init__(
        self,
        collection: str = "hud_memory",
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        """Initialize SessionMemoryTool.

        Args:
            collection: Qdrant collection name
            qdrant_url: Qdrant server URL (enables vector search)
            qdrant_api_key: Qdrant API key
        """
        super().__init__()
        self.name = "memory"
        self.title = "Memory"
        self.description = "Add and search session memory"
        self._backend = self._build_backend(collection, qdrant_url, qdrant_api_key)

    def _build_backend(
        self,
        collection: str,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
    ) -> Any:
        """Build the appropriate backend."""
        if qdrant_url:
            try:
                from qdrant_client import QdrantClient  # type: ignore[import-not-found]
                from qdrant_client.http.models import (  # type: ignore[import-not-found]
                    Distance,
                    VectorParams,
                )
            except ImportError:
                LOGGER.warning("Qdrant is not installed, using in-memory store")
                return self  # Use self as backend (BaseSessionMemoryTool)

            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            try:
                client.get_collection(collection)
            except Exception:
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
            return _QdrantBackend(client, collection)

        return self  # Use self as backend (BaseSessionMemoryTool)

    @property
    def parameters(self) -> dict[str, Any]:  # type: ignore[override]
        """Tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "search"],
                    "description": "add = store text, search = retrieve similar items",
                },
                "text": {
                    "type": "string",
                    "description": "Content to store or query",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to store with the entry",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                    "description": "Number of results to return when searching",
                },
            },
            "required": ["action", "text"],
        }

    async def __call__(
        self,
        action: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[ContentBlock]:
        """Execute memory action.

        Args:
            action: "add" or "search"
            text: Content to store or query
            metadata: Optional metadata for add action
            top_k: Number of results for search action

        Returns:
            List of ContentBlocks with result
        """
        if action == "add":
            if self._backend is self:
                self.add_entry(text=text, metadata=metadata)
            else:
                self._backend.add(text=text, metadata=metadata)
            return [TextContent(text="stored", type="text")]

        if action == "search":
            if self._backend is self:
                entries = self.search_entries(query=text, top_k=top_k)
            else:
                entries = self._backend.query(query=text, top_k=top_k)

            if not entries:
                return [TextContent(text="no matches", type="text")]

            lines = []
            for idx, entry in enumerate(entries, 1):
                meta = entry.metadata or {}
                meta_str = f" | metadata={meta}" if meta else ""
                lines.append(f"{idx}. {entry.text}{meta_str}")
            return [TextContent(text="\n".join(lines), type="text")]

        return [TextContent(text="unknown action", type="text")]


class _QdrantBackend:
    """Qdrant wrapper with sentence-transformer embeddings."""

    def __init__(self, client: Any, collection: str) -> None:
        self.client = client
        self.collection = collection
        self._embedder = self._load_embedder()

    def _load_embedder(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("sentence-transformers is required for Qdrant backend") from e
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Add an entry to Qdrant."""
        vec = self._embedder.encode(text).tolist()
        payload = {"text": text, "metadata": metadata or {}}
        self.client.upsert(
            collection_name=self.collection,
            points=[{"id": uuid.uuid4().hex, "vector": vec, "payload": payload}],
        )

    def query(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search Qdrant for similar entries."""
        vec = self._embedder.encode(query).tolist()
        res = self.client.search(
            collection_name=self.collection,
            query_vector=vec,
            limit=top_k,
            with_payload=True,
        )
        entries: list[MemoryEntry] = []
        for point in res:
            payload = point.payload or {}
            entries.append(
                MemoryEntry(
                    text=payload.get("text", ""),
                    metadata=payload.get("metadata", {}),
                    tokens=set(),
                )
            )
        return entries


# Backwards compatibility alias
MemoryTool = SessionMemoryTool

__all__ = ["MemoryTool", "SessionMemoryTool"]
