"""Lightweight memory tool with optional Qdrant backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp.types import ContentBlock, TextContent

from hud.tools.base import BaseTool


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in text.split() if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class MemoryEntry:
    text: str
    metadata: dict[str, Any]
    tokens: set[str]


class InMemoryStore:
    """Simple token-overlap store."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._entries.append(
            MemoryEntry(text=text, metadata=metadata or {}, tokens=_tokenize(text))
        )

    def query(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        q_tokens = _tokenize(query)
        scored = [(entry, _jaccard(q_tokens, entry.tokens)) for entry in self._entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in scored[:top_k] if score > 0.0]


class MemoryTool(BaseTool):
    """Add and search short-term memory for a session.

    If Qdrant is available and configured (QDRANT_URL), a remote collection is used.
    Otherwise, an in-memory fallback is used.
    """

    def __init__(
        self,
        collection: str = "hud_memory",
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._backend = self._build_backend(collection, qdrant_url, qdrant_api_key)

    def _build_backend(
        self, collection: str, qdrant_url: str | None, qdrant_api_key: str | None
    ) -> Any:
        if qdrant_url:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http.models import Distance, VectorParams
            except Exception:
                pass
            else:
                client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                try:
                    client.get_collection(collection)
                except Exception:
                    client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                    )
                return QdrantBackend(client, collection)
        return InMemoryStore()

    @property
    def parameters(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "search"],
                    "description": "add = store text, search = retrieve similar items",
                },
                "text": {"type": "string", "description": "content to store or query"},
                "metadata": {
                    "type": "object",
                    "description": "optional metadata to store with the entry",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                    "description": "results to return when searching",
                },
            },
            "required": ["action", "text"],
        }

    async def __call__(
        self, action: str, text: str, metadata: dict[str, Any] | None = None, top_k: int = 5
    ) -> list[ContentBlock]:
        if action == "add":
            self._backend.add(text=text, metadata=metadata)
            return [TextContent(text="stored", type="text")]
        if action == "search":
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


class QdrantBackend:
    """Minimal Qdrant wrapper with on-the-fly sentence-transformer embeddings."""

    def __init__(self, client: Any, collection: str) -> None:
        self.client = client
        self.collection = collection
        self._embedder = self._load_embedder()

    def _load_embedder(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("sentence-transformers is required for Qdrant backend") from e
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        vec = self._embedder.encode(text).tolist()
        payload = {"text": text, "metadata": metadata or {}}
        self.client.upsert(
            collection_name=self.collection,
            points=[{"vector": vec, "payload": payload}],
        )

    def query(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
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
