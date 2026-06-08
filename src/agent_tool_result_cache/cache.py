"""Core ToolResultCache implementation."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def _hash_args(tool_name: str, args: dict[str, Any]) -> str:
    """Canonical hash of (tool_name, args). Sort keys for deterministic bytes."""
    payload = json.dumps(
        {"tool": tool_name, "args": args},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class _Entry:
    """One cache row. expires_at is wall-clock epoch seconds, or None for never-expires."""

    value: Any
    expires_at: float | None  # monotonic-time deadline; None means never expires
    expires_at_wall: float | None  # wall-clock deadline (for persistence)
    tool: str


@dataclass
class _ToolStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0


class ToolResultCache:
    """In-process LRU+TTL cache for tool-call results.

    Tools (web fetch, DB query, search, etc.) are often expensive in
    dollars or wall time but idempotent within a window. This cache
    stores their results by `(tool_name, args)` with a per-tool TTL.

    Per-tool TTL override:
      * `default_ttl_seconds` covers tools you have not overridden.
      * `set_ttl(tool_name, ttl_seconds)` sets the TTL for one tool.
      * `ttl_seconds=0` means never cache (set is a no-op, get always misses).

    Optional persistence:
      * If `persist_path` is set, every `set` appends one JSONL line and
        the cache loads on init (expired rows are dropped).

    Thread-safe via an internal `threading.RLock`. Async-friendly via
    `cached_async`; the lock is released during the wrapped coroutine call.
    """

    def __init__(
        self,
        default_ttl_seconds: float = 600.0,
        maxsize: int = 10_000,
        persist_path: str | Path | None = None,
    ) -> None:
        if default_ttl_seconds < 0:
            raise ValueError("default_ttl_seconds must be >= 0")
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self._default_ttl = float(default_ttl_seconds)
        self._maxsize = int(maxsize)
        self._persist_path = Path(persist_path) if persist_path is not None else None
        self._ttls: dict[str, float] = {}
        self._data: OrderedDict[str, _Entry] = OrderedDict()
        self._stats_total = _ToolStats()
        self._stats_by_tool: dict[str, _ToolStats] = {}
        self._lock = threading.RLock()

        if self._persist_path is not None:
            self._load_from_disk()

    # ---- configuration ----

    def set_ttl(self, tool_name: str, ttl_seconds: float) -> None:
        """Override TTL for a specific tool. ttl_seconds=0 disables caching for it."""
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")
        with self._lock:
            self._ttls[tool_name] = float(ttl_seconds)

    def get_ttl(self, tool_name: str) -> float:
        """Return the effective TTL for a tool. Falls back to the default."""
        with self._lock:
            return self._ttls.get(tool_name, self._default_ttl)

    # ---- core get/set ----

    def set(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        ttl_seconds: float | None = None,
    ) -> None:
        """Cache `result` for `(tool_name, args)`. If the effective TTL is 0,
        this is a no-op (the tool is opted out of caching)."""
        ttl = (
            ttl_seconds
            if ttl_seconds is not None
            else self._ttls.get(tool_name, self._default_ttl)
        )
        if ttl <= 0:
            return  # never cache

        key = _hash_args(tool_name, args)
        now_mono = time.monotonic()
        now_wall = time.time()
        entry = _Entry(
            value=result,
            expires_at=now_mono + ttl,
            expires_at_wall=now_wall + ttl,
            tool=tool_name,
        )
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = entry
            self._evict_if_needed()
            if self._persist_path is not None:
                self._append_to_disk(tool_name, key, result, entry.expires_at_wall)

    def get(self, tool_name: str, args: dict[str, Any]) -> Any | None:
        """Return cached value for `(tool_name, args)` or None on miss/expired.

        Bumps hit/miss stats. If you stored an actual `None` value, use
        `contains()` first to disambiguate it from a miss.
        """
        found, value = self._lookup(tool_name, args)
        return value if found else None

    def _lookup(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, Any]:
        """Internal: returns (found, value). Bumps stats."""
        key = _hash_args(tool_name, args)
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            tool_stats = self._tool_stats(tool_name)
            if entry is None:
                tool_stats.misses += 1
                self._stats_total.misses += 1
                return (False, None)
            if entry.expires_at is not None and entry.expires_at <= now:
                # expired: drop + count
                self._data.pop(key, None)
                tool_stats.expirations += 1
                tool_stats.misses += 1
                self._stats_total.expirations += 1
                self._stats_total.misses += 1
                return (False, None)
            # hit: bump LRU
            self._data.move_to_end(key)
            tool_stats.hits += 1
            self._stats_total.hits += 1
            return (True, entry.value)

    def contains(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Return True if a non-expired entry exists, without LRU bump or stats touch."""
        key = _hash_args(tool_name, args)
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            return not (entry.expires_at is not None and entry.expires_at <= now)

    def delete(self, tool_name: str, args: dict[str, Any] | None = None) -> int:
        """Delete one entry, or all entries for the tool when args is None.
        Returns the number of entries removed."""
        with self._lock:
            if args is not None:
                key = _hash_args(tool_name, args)
                return 1 if self._data.pop(key, None) is not None else 0
            # drop all entries for this tool
            removed = 0
            for k in list(self._data.keys()):
                if self._data[k].tool == tool_name:
                    self._data.pop(k, None)
                    removed += 1
            return removed

    def invalidate(self, tool_name: str | None = None) -> int:
        """Invalidate all entries (tool_name=None) or all entries for one tool.
        Returns the number of entries removed."""
        with self._lock:
            if tool_name is None:
                removed = len(self._data)
                self._data.clear()
                return removed
            return self.delete(tool_name)

    # ---- decorators ----

    def cached(self, tool_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator: cache results of a sync function under `tool_name`.

        Cache key is derived from the function's bound arguments (kwargs +
        positional-by-name), hashed with the tool name.
        """

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            import functools
            import inspect

            sig = inspect.signature(fn)

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                key_args = _bind_to_dict(sig, args, kwargs)
                found, value = self._lookup(tool_name, key_args)
                if found:
                    return value  # type: ignore[no-any-return]
                result = fn(*args, **kwargs)
                self.set(tool_name, key_args, result)
                return result

            return wrapper

        return decorator

    def cached_async(
        self, tool_name: str
    ) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
        """Decorator: cache results of an async function under `tool_name`."""

        def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            import functools
            import inspect

            sig = inspect.signature(fn)

            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                key_args = _bind_to_dict(sig, args, kwargs)
                found, value = self._lookup(tool_name, key_args)
                if found:
                    return value  # type: ignore[no-any-return]
                result = await fn(*args, **kwargs)
                self.set(tool_name, key_args, result)
                return result

            return wrapper

        return decorator

    # ---- introspection ----

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def stats(self) -> dict[str, Any]:
        """Snapshot of hits/misses/evictions/expirations per tool and total."""
        with self._lock:
            per_tool = {
                tool: {
                    "hits": s.hits,
                    "misses": s.misses,
                    "evictions": s.evictions,
                    "expirations": s.expirations,
                }
                for tool, s in self._stats_by_tool.items()
            }
            return {
                "hits": self._stats_total.hits,
                "misses": self._stats_total.misses,
                "evictions": self._stats_total.evictions,
                "expirations": self._stats_total.expirations,
                "size": len(self._data),
                "per_tool": per_tool,
            }

    # ---- internal ----

    def _tool_stats(self, tool_name: str) -> _ToolStats:
        s = self._stats_by_tool.get(tool_name)
        if s is None:
            s = _ToolStats()
            self._stats_by_tool[tool_name] = s
        return s

    def _evict_if_needed(self) -> None:
        """Caller must hold the lock."""
        while len(self._data) > self._maxsize:
            evicted_key, evicted_entry = self._data.popitem(last=False)
            del evicted_key
            ts = self._tool_stats(evicted_entry.tool)
            ts.evictions += 1
            self._stats_total.evictions += 1

    # ---- persistence ----

    def _append_to_disk(
        self, tool: str, key_hash: str, value: Any, expires_at_wall: float | None
    ) -> None:
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "tool": tool,
            "key_hash": key_hash,
            "value_json": json.dumps(value, default=str),
            "expires_at": expires_at_wall,
        }
        with self._persist_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _load_from_disk(self) -> None:
        """Read persisted JSONL, drop expired rows, replay live rows."""
        if self._persist_path is None or not self._persist_path.exists():
            return
        now_wall = time.time()
        now_mono = time.monotonic()
        with self._persist_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                expires_at_wall = row.get("expires_at")
                if expires_at_wall is None or expires_at_wall <= now_wall:
                    continue  # expired or no-expiry-on-disk: drop
                value = json.loads(row["value_json"])
                ttl_left = expires_at_wall - now_wall
                entry = _Entry(
                    value=value,
                    expires_at=now_mono + ttl_left,
                    expires_at_wall=expires_at_wall,
                    tool=row["tool"],
                )
                # latest write wins because JSONL is append-only and ordered
                self._data[row["key_hash"]] = entry
                self._data.move_to_end(row["key_hash"])
        # post-load eviction in case the file held more than maxsize
        self._evict_if_needed()


def _bind_to_dict(
    sig: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Bind call args to the signature and return a name->value dict for hashing."""
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)
