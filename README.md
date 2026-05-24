# agent-tool-result-cache

[![PyPI](https://img.shields.io/pypi/v/agent-tool-result-cache.svg)](https://pypi.org/project/agent-tool-result-cache/)
[![Python](https://img.shields.io/pypi/pyversions/agent-tool-result-cache.svg)](https://pypi.org/project/agent-tool-result-cache/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**In-process LRU+TTL cache for TOOL call results.**

When an agent calls a tool (`web_fetch`, `db_query`, `search_docs`,
etc.), the call is often expensive in dollars or wall time but
idempotent within a window. This library caches those tool results
by `(tool_name, args)` with a per-tool TTL, so the same arguments
inside the window return instantly without firing the real tool.

Different scope from LLM-response caches:

* Use `llm-cache-mem` for caching LLM responses.
* Use `agent-tool-result-cache` (this lib) for caching tool outputs.

## Install

```bash
pip install agent-tool-result-cache
```

## Basic

```python
from agent_tool_result_cache import ToolResultCache

cache = ToolResultCache(default_ttl_seconds=600)

cache.set("search", {"query": "foo"}, result=["a", "b"])
hit = cache.get("search", {"query": "foo"})  # ["a", "b"] or None on miss
```

## Per-tool TTL

```python
cache = ToolResultCache(default_ttl_seconds=600)
cache.set_ttl("get_weather", ttl_seconds=300)
cache.set_ttl("db_query_slow", ttl_seconds=5)
cache.set_ttl("get_current_time", ttl_seconds=0)  # never cache
```

`ttl_seconds=0` opts a tool out of caching entirely. `set()` becomes a
no-op for that tool and every `get()` returns `None`.

## Decorator

```python
@cache.cached("search")
def search(query: str, limit: int = 10) -> list:
    # real call
    ...

search("LLM agent libraries")           # miss, fires real call
search("LLM agent libraries")           # hit, no real call
search("LLM agent libraries", limit=5)  # miss, different args
```

Cache key is derived from the function's bound arguments, sorted and
hashed with sha256 alongside the tool name. Argument dict order does
not matter.

## Async

```python
@cache.cached_async("fetch_url")
async def fetch_url(url: str) -> str:
    ...
```

## Persistence

Pass `persist_path` to survive process restarts. The cache appends one
JSONL row per `set()` and replays the file on init, dropping expired
rows. Values must be JSON-serializable when persistence is on.

```python
cache = ToolResultCache(persist_path="~/.cache/my-agent/tools.jsonl")
```

## Stats

```python
cache.stats()
# {
#   "hits": 12, "misses": 3, "evictions": 0, "expirations": 1,
#   "size": 8,
#   "per_tool": {
#     "search":      {"hits": 10, "misses": 1, "evictions": 0, "expirations": 0},
#     "get_weather": {"hits": 2,  "misses": 2, "evictions": 0, "expirations": 1},
#   },
# }
```

## What it does NOT do

* No network. Doesn't fetch anything; you wrap your existing tool.
* No serialization for non-JSON values in persistence mode. Values
  must be JSON-serializable when `persist_path` is set.
* No cross-process sharing. The cache is in-process. For multi-host
  agents, wrap a Redis or DB lookup instead.
* No partial-key invalidation. Use `invalidate(tool_name)` to drop a
  whole tool's entries, or `delete(tool_name, args)` for one row.

## Siblings

This is the tool-result cache. The agent stack ships related libs:

* [`llm-cache-mem`](https://pypi.org/project/llm-cache-mem/): caches
  LLM responses (different scope: model requests, not tool calls).
* [`llm-batch-coalesce`](https://pypi.org/project/llm-batch-coalesce/):
  in-flight dedup of identical concurrent LLM calls.
* [`llm-message-hash-py`](https://pypi.org/project/llm-message-hash-py/):
  canonical hash of an LLM request structure (used inside the libs above).

## License

MIT
