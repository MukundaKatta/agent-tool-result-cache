"""agent-tool-result-cache: LRU+TTL cache for agent tool call results.

The public surface is :class:`ToolResultCache` (and the :class:`CacheEntry`
dataclass it stores). The cache is keyed by a stable SHA-256 hash of the
``(tool_name, args)`` pair, so equal arguments always map to the same entry
regardless of dict ordering.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__version__ = "0.1.0"


def _hash_call(tool_name: str, args: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest for a ``(tool_name, args)`` call.

    Keys are sorted so two equal-but-differently-ordered ``args`` dicts hash
    identically. ``default=str`` keeps the hash from raising on values that are
    not natively JSON-serializable (datetimes, ``Path``, etc.); such values
    are compared by their string form.
    """
    raw = json.dumps(
        {"tool": tool_name, "args": args},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class CacheEntry:
    """A single cached tool result.

    Attributes:
        key: SHA-256 hash of the originating ``(tool_name, args)`` pair.
        result: The cached return value of the tool call.
        tool_name: Name of the tool that produced this result. Stored so the
            cache can invalidate every entry belonging to a tool.
        created_at: ``time.monotonic()`` timestamp captured when the entry was
            created. Monotonic time is used so the TTL is unaffected by
            wall-clock changes (NTP steps, DST, manual clock edits).
        ttl: Time-to-live in seconds, or ``None`` for an entry that never
            expires.
        hit_count: Number of times this entry has been returned from
            :meth:`ToolResultCache.get`.
    """

    key: str
    result: Any
    tool_name: str = ""
    created_at: float = field(default_factory=time.monotonic)
    ttl: Optional[float] = None
    hit_count: int = 0

    @property
    def expired(self) -> bool:
        """True if this entry has a TTL and that TTL has elapsed."""
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
        """Create a cache.

        Args:
            max_size: Maximum number of live entries. Once full, inserting a
                new key evicts the least-recently-used entry. Must be >= 1.
            default_ttl: Default time-to-live in seconds applied to entries
                that do not specify their own ``ttl``. ``None`` means entries
                never expire by default.

        Raises:
            ValueError: If ``max_size`` is less than 1.
        """
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _evict_if_needed(self) -> None:
        """Evict least-recently-used entries until there is room for one more."""
        while len(self._store) >= self._max_size:
            self._store.popitem(last=False)

    def put(self, tool_name: str, args: dict[str, Any], result: Any, ttl: Optional[float] = None) -> str:
        """Store ``result`` for ``(tool_name, args)`` and return the cache key.

        Inserting (or overwriting) an entry marks it as the most-recently-used.
        If the cache is full, the least-recently-used entry is evicted first.

        Args:
            tool_name: Logical name of the tool whose result is being cached.
            args: Keyword arguments that identify this specific call. Must be
                JSON-serializable.
            result: The value to cache.
            ttl: Per-entry time-to-live in seconds. Falls back to the cache's
                ``default_ttl`` when ``None``.

        Returns:
            The SHA-256 cache key for the stored entry.
        """
        key = _hash_call(tool_name, args)
        # Only evict to make room when inserting a genuinely new key;
        # overwriting an existing key does not grow the store.
        if key not in self._store:
            self._evict_if_needed()
        entry = CacheEntry(
            key=key,
            result=result,
            tool_name=tool_name,
            ttl=ttl if ttl is not None else self._default_ttl,
        )
        self._store[key] = entry
        self._store.move_to_end(key)
        return key

    def get(self, tool_name: str, args: dict[str, Any]) -> Optional[Any]:
        """Return the cached result for ``(tool_name, args)`` or ``None``.

        A hit moves the entry to most-recently-used and bumps the hit
        counters. Expired entries are evicted lazily and counted as a miss.

        Note:
            ``None`` is returned both for a miss and for a stored value of
            ``None``. Use :meth:`has` if you need to distinguish the two.
        """
        key = _hash_call(tool_name, args)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.expired:
            del self._store[key]
            self._misses += 1
            return None
        entry.hit_count += 1
        self._store.move_to_end(key)
        self._hits += 1
        return entry.result

    def has(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Return ``True`` if a live (non-expired) entry exists.

        Unlike :meth:`get`, this does not affect LRU order or hit/miss stats,
        but it does lazily evict an entry that is found to be expired.
        """
        key = _hash_call(tool_name, args)
        entry = self._store.get(key)
        if entry is None:
            return False
        if entry.expired:
            del self._store[key]
            return False
        return True

    def invalidate(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Remove a single entry by ``(tool_name, args)``.

        Returns:
            ``True`` if an entry was removed, ``False`` if it was absent.
        """
        key = _hash_call(tool_name, args)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def invalidate_tool(self, tool_name: str) -> int:
        """Remove every cached result belonging to ``tool_name``.

        Each entry records the tool that produced it, so this filters the
        store by that name and drops the matches.

        Args:
            tool_name: Name of the tool whose entries should be removed.

        Returns:
            The number of entries removed.
        """
        to_remove = [
            key for key, entry in self._store.items() if entry.tool_name == tool_name
        ]
        for key in to_remove:
            del self._store[key]
        return len(to_remove)

    def prune_expired(self) -> int:
        """Eagerly drop every expired entry.

        Entries normally expire lazily on access; call this to reclaim memory
        proactively (for example on a timer).

        Returns:
            The number of expired entries removed.
        """
        expired = [k for k, e in self._store.items() if e.expired]
        for k in expired:
            del self._store[k]
        return len(expired)

    def clear(self) -> None:
        """Remove all entries and reset the hit/miss counters."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Current number of stored entries (including not-yet-pruned expired ones)."""
        return len(self._store)

    @property
    def hits(self) -> int:
        """Total number of cache hits since creation or the last :meth:`clear`."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total number of cache misses since creation or the last :meth:`clear`."""
        return self._misses

    @property
    def hit_rate(self) -> Optional[float]:
        """Hits divided by total lookups, or ``None`` if there have been none."""
        total = self._hits + self._misses
        if total == 0:
            return None
        return self._hits / total

    @property
    def stats(self) -> dict[str, Any]:
        """Snapshot of ``size``, ``hits``, ``misses`` and ``hit_rate``."""
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }

    def wrap(self, tool_name: str, ttl: Optional[float] = None) -> Callable[..., Any]:
        """Decorator that caches a function's results keyed by its kwargs.

        The wrapped function must be called with keyword arguments only — the
        kwargs form the cache key, so positional calls would not be cached
        consistently.

        Args:
            tool_name: Name under which results are cached.
            ttl: Per-entry TTL in seconds; falls back to ``default_ttl``.

        Example::

            cache = ToolResultCache()

            @cache.wrap("search_web")
            def search_web(query: str) -> list:
                ...

            search_web(query="python")  # executes
            search_web(query="python")  # served from cache
        """
        import functools

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            def wrapper(**kwargs: Any) -> Any:
                # has() distinguishes a cached None from a miss, so functions
                # that legitimately return None are still cached correctly.
                if self.has(tool_name, kwargs):
                    return self.get(tool_name, kwargs)
                result = fn(**kwargs)
                self.put(tool_name, kwargs, result, ttl=ttl)
                return result
            return wrapper
        return decorator


__all__ = ["ToolResultCache", "CacheEntry", "__version__"]
