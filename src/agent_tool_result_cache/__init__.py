"""
agent-tool-result-cache: LRU+TTL cache for agent tool call results.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Sentinel distinguishing "no cached value" from a cached value that is None.
_MISS = object()


def _hash_call(tool_name: str, args: dict[str, Any]) -> str:
    raw = json.dumps(
        {"tool": tool_name, "args": args}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class CacheEntry:
    key: str
    result: Any
    tool_name: str = ""
    created_at: float = field(default_factory=time.monotonic)
    ttl: Optional[float] = None
    hit_count: int = 0

    @property
    def expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.monotonic() - self.created_at) >= self.ttl


class ToolResultCache:
    """
    LRU + TTL cache for tool call results.

    Usage::

        cache = ToolResultCache(max_size=128, default_ttl=300.0)

        @cache.wrap("search_web")
        def search_web(query: str, limit: int = 5) -> list:
            ...

        results = search_web(query="python", limit=5)   # cached on 2nd call
    """

    def __init__(
        self,
        max_size: int = 256,
        default_ttl: Optional[float] = None,
    ) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _evict_if_needed(self) -> None:
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def put(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        ttl: Optional[float] = None,
    ) -> str:
        key = _hash_call(tool_name, args)
        entry = CacheEntry(
            key=key,
            result=result,
            tool_name=tool_name,
            ttl=ttl if ttl is not None else self._default_ttl,
        )
        # Insert/overwrite first, then evict. Evicting beforehand could drop a
        # live entry when we are only updating an existing key (no size growth).
        self._store[key] = entry
        self._store.move_to_end(key)
        self._evict_if_needed()
        return key

    def _get_or_sentinel(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Like :meth:`get` but returns ``_MISS`` on miss/expiry so a cached
        ``None`` can be distinguished from a cache miss. Updates stats/LRU."""
        key = _hash_call(tool_name, args)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return _MISS
        if entry.expired:
            del self._store[key]
            self._misses += 1
            return _MISS
        entry.hit_count += 1
        self._store.move_to_end(key)
        self._hits += 1
        return entry.result

    def get(self, tool_name: str, args: dict[str, Any]) -> Optional[Any]:
        result = self._get_or_sentinel(tool_name, args)
        return None if result is _MISS else result

    def has(self, tool_name: str, args: dict[str, Any]) -> bool:
        key = _hash_call(tool_name, args)
        entry = self._store.get(key)
        if entry is None:
            return False
        if entry.expired:
            del self._store[key]
            return False
        return True

    def invalidate(self, tool_name: str, args: dict[str, Any]) -> bool:
        key = _hash_call(tool_name, args)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def invalidate_tool(self, tool_name: str) -> int:
        """Remove all cached results for a given tool. Returns the count removed."""
        to_remove = [
            key for key, entry in self._store.items() if entry.tool_name == tool_name
        ]
        for key in to_remove:
            del self._store[key]
        return len(to_remove)

    def prune_expired(self) -> int:
        expired = [k for k, e in self._store.items() if e.expired]
        for k in expired:
            del self._store[k]
        return len(expired)

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> Optional[float]:
        total = self._hits + self._misses
        if total == 0:
            return None
        return self._hits / total

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }

    def wrap(self, tool_name: str, ttl: Optional[float] = None) -> Callable[..., Any]:
        """
        Decorator: cache results of the wrapped function by its kwargs.
        The function must accept keyword arguments only (or at least the ones
        that determine uniqueness).
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            def wrapper(**kwargs: Any) -> Any:
                # Use a sentinel so a cached result of ``None`` is still served
                # from the cache instead of triggering a re-run on every call.
                cached = self._get_or_sentinel(tool_name, kwargs)
                if cached is not _MISS:
                    return cached
                result = fn(**kwargs)
                self.put(tool_name, kwargs, result, ttl=ttl)
                return result

            return wrapper

        return decorator


__all__ = ["ToolResultCache", "CacheEntry"]
