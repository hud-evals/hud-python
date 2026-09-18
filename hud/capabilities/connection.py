"""Controller-provided HTTP connections bound to trusted process executions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from collections.abc import Mapping

_NAME = re.compile(r"[a-z][a-z0-9_-]{0,62}")
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")


@dataclass(frozen=True, slots=True)
class Connection:
    """An authenticated HTTP endpoint available only to one bound process.

    ``url`` and ``headers`` describe the controller-selected upstream. The
    process receives :attr:`client_url` instead and never receives the header
    values. A runtime which cannot enforce process-bound connections must
    reject the connection rather than expose it workspace-wide.
    """

    name: str
    capability: str
    url: str
    headers: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(
                "connection name must start with a lowercase letter and contain only "
                "lowercase letters, digits, underscores, or hyphens"
            )
        if not self.capability or self.capability.strip() != self.capability:
            raise ValueError("connection capability must not be empty or padded")
        parts = urlsplit(self.url)
        if parts.scheme not in {"http", "https"} or parts.hostname is None:
            raise ValueError("connection url must be an HTTP(S) URL with a hostname")
        if parts.username is not None or parts.password is not None:
            raise ValueError("connection url must not contain credentials")
        if parts.query or parts.fragment:
            raise ValueError("connection url must not contain a query or fragment")
        if not self.headers:
            raise ValueError("connection headers must not be empty")
        for name, value in self.headers.items():
            if not _HEADER_NAME.fullmatch(name):
                raise ValueError(f"invalid connection header name: {name!r}")
            if not value or any(character in value for character in "\r\n"):
                raise ValueError(f"invalid connection header value for {name!r}")
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))

    @property
    def host(self) -> str:
        """Private hostname installed only inside the selected workspace."""
        digest = hashlib.sha256(f"{self.name}\0{self.url}".encode()).hexdigest()[:12]
        return f"{self.name}-{digest}.hud.invalid"

    @property
    def port(self) -> int:
        return 80

    @property
    def client_url(self) -> str:
        """Credential-free URL configured in the bound process."""
        parts = urlsplit(self.url)
        return urlunsplit(("http", self.host, parts.path, "", ""))

    def to_wire(self) -> dict[str, object]:
        return {
            "name": self.name,
            "capability": self.capability,
            "url": self.url,
            "headers": dict(self.headers),
        }

    @classmethod
    def from_wire(cls, value: object) -> Connection:
        if not isinstance(value, dict):
            raise ValueError("connections must be objects")
        name = value.get("name")
        capability = value.get("capability")
        url = value.get("url")
        raw_headers = value.get("headers")
        if not all(isinstance(item, str) for item in (name, capability, url)):
            raise ValueError("connection name, capability, and url must be strings")
        if not isinstance(raw_headers, dict) or not all(
            isinstance(key, str) and isinstance(header_value, str)
            for key, header_value in raw_headers.items()
        ):
            raise ValueError("connection headers must map strings to strings")
        assert isinstance(name, str) and isinstance(capability, str) and isinstance(url, str)
        return cls(
            name=name,
            capability=capability,
            url=url,
            headers=dict(cast("dict[str, str]", raw_headers)),
        )


__all__ = ["Connection"]
