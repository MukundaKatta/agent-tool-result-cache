"""agent-tool-result-cache - in-process LRU+TTL cache for tool-call results.

Caches the result of tool calls (web fetch, DB query, search, etc.)
by `(tool_name, args)` with a per-tool TTL. Different scope from
LLM-response caches: tools may be expensive in dollars or wall time,
but their results are usually idempotent within a window.

    from agent_tool_result_cache import ToolResultCache

    cache = ToolResultCache(default_ttl_seconds=600)
    cache.set_ttl("get_weather", ttl_seconds=300)
    cache.set_ttl("get_current_time", ttl_seconds=0)  # never cache

    @cache.cached("search")
    def search(query: str) -> list:
        ...

    results = search("LLM agent libraries")  # miss, fires real
    results = search("LLM agent libraries")  # hit, returns cached

Direct API:

    cache.set("search", {"query": "foo"}, result=["a", "b"])
    hit = cache.get("search", {"query": "foo"})  # ["a", "b"] or None

Async:

    @cache.cached_async("fetch_url")
    async def fetch_url(url: str) -> str: ...

Optional persistence:

    cache = ToolResultCache(persist_path="~/.cache/tools.jsonl")

Sibling libs:

  * llm-cache-mem - caches LLM responses, not tool results.
  * llm-batch-coalesce - in-flight dedup of identical inflight LLM calls.
  * llm-message-hash-py - canonical hash of an LLM request structure.
"""

from agent_tool_result_cache.cache import ToolResultCache

__version__ = "0.1.0"

__all__ = [
    "ToolResultCache",
    "__version__",
]
