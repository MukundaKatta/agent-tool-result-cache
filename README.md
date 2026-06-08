# agent-tool-result-cache

An in-process **LRU + TTL cache for agent tool-call results**.

LLM agents call tools (web search, DB queries, HTTP fetches, file reads) that
are often expensive in latency or dollars but **idempotent within a short
window**. This library memoizes those results keyed by `(tool_name, args)`,
so repeated calls with the same arguments are served from memory instead of
re-running the tool.

- **Zero dependencies** — standard library only.
- **Python 3.10+**.
- **Bounded** — least-recently-used eviction once `max_size` is reached.
- **Expiring** — per-cache default TTL plus per-entry TTL overrides.
- **Order-stable keys** — `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` hash to
  the same entry.
- **Observable** — hit/miss counters and a `hit_rate`.

## Install

From source (PyPI publishing is not set up yet):

```bash
git clone https://github.com/MukundaKatta/agent-tool-result-cache.git
cd agent-tool-result-cache
pip install .
```

## Quick start

```python
from agent_tool_result_cache import ToolResultCache

cache = ToolResultCache(max_size=256, default_ttl=300.0)  # 5-minute TTL

# Manual put / get -----------------------------------------------------------
cache.put("search_web", {"query": "python", "limit": 5}, ["doc-1", "doc-2"])

hit = cache.get("search_web", {"query": "python", "limit": 5})
print(hit)            # ['doc-1', 'doc-2']

miss = cache.get("search_web", {"query": "rust", "limit": 5})
print(miss)           # None

print(cache.stats)    # {'size': 1, 'hits': 1, 'misses': 1, 'hit_rate': 0.5}
```

### Decorate a tool function

`@cache.wrap` turns any keyword-only-called function into a cached one. The
keyword arguments form the cache key, so identical calls are only executed
once until the entry expires.

```python
from agent_tool_result_cache import ToolResultCache

cache = ToolResultCache(default_ttl=60.0)

@cache.wrap("search_web", ttl=120.0)  # per-tool TTL override
def search_web(query: str, limit: int = 5) -> list[str]:
    print(f"actually searching for {query!r}...")
    return [f"{query}-result-{i}" for i in range(limit)]

search_web(query="python")   # prints "actually searching..." and runs
search_web(query="python")   # served from cache, no print
```

Functions that legitimately return `None` are cached correctly: `@cache.wrap`
uses `has()` internally, so a cached `None` is not mistaken for a miss.

### Invalidation and TTL

```python
cache = ToolResultCache(default_ttl=300.0)

cache.put("db_query", {"sql": "SELECT 1"}, 1, ttl=10.0)  # per-entry TTL

cache.invalidate("db_query", {"sql": "SELECT 1"})  # drop one entry -> True
cache.invalidate_tool("db_query")                  # drop every db_query entry
cache.prune_expired()                              # sweep all expired entries
cache.clear()                                      # drop everything + reset stats
```

## API reference

### `ToolResultCache(max_size=256, default_ttl=None)`

| Argument      | Type              | Description                                              |
| ------------- | ----------------- | ------------------------------------------------------- |
| `max_size`    | `int`             | Max live entries before LRU eviction kicks in (`>= 1`). |
| `default_ttl` | `float \| None`   | Default time-to-live in seconds; `None` never expires.  |

#### Methods

| Method | Returns | Description |
| ------ | ------- | ----------- |
| `put(tool_name, args, result, ttl=None)` | `str` | Cache `result`; returns the SHA-256 key. |
| `get(tool_name, args)` | `Any \| None` | Cached value or `None` on miss/expiry (bumps stats). |
| `has(tool_name, args)` | `bool` | Live entry exists? No LRU/stats side effects. |
| `invalidate(tool_name, args)` | `bool` | Remove one entry; `True` if it existed. |
| `invalidate_tool(tool_name)` | `int` | Remove every entry for a tool; returns the count. |
| `prune_expired()` | `int` | Eagerly drop expired entries; returns the count. |
| `clear()` | `None` | Remove everything and reset hit/miss counters. |
| `wrap(tool_name, ttl=None)` | decorator | Cache a keyword-called function's results. |

#### Properties

| Property | Type | Description |
| -------- | ---- | ----------- |
| `size` | `int` | Number of stored entries. |
| `hits` | `int` | Total cache hits. |
| `misses` | `int` | Total cache misses. |
| `hit_rate` | `float \| None` | `hits / (hits + misses)`, or `None` if no lookups yet. |
| `stats` | `dict` | Snapshot of `size`, `hits`, `misses`, `hit_rate`. |

### `CacheEntry`

A dataclass describing one cached row: `key`, `result`, `tool_name`,
`created_at`, `ttl`, `hit_count`, and an `expired` property. You normally do
not construct these directly.

## Notes on semantics

- **Time source.** TTLs are measured with `time.monotonic()`, so they are
  unaffected by wall-clock changes (NTP steps, DST, manual edits).
- **`None` values.** `get()` returns `None` for both a miss and a stored
  `None`. Use `has()` when you need to tell them apart.
- **Key hashing.** Keys are a SHA-256 of the JSON-serialized
  `(tool_name, args)` with sorted keys. Values that are not natively
  JSON-serializable are coerced via `str()` for hashing purposes.

## Development

Run the test suite (standard-library `unittest`, no third-party deps):

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

CI runs the same suite on Python 3.10–3.13 and verifies the package installs
and imports.

## License

MIT — see [LICENSE](LICENSE).
