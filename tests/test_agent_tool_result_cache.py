"""Tests for agent-tool-result-cache."""
import time
import pytest
from agent_tool_result_cache import ToolResultCache, CacheEntry


def test_put_and_get():
    cache = ToolResultCache()
    cache.put("search", {"query": "python"}, ["result1", "result2"])
    result = cache.get("search", {"query": "python"})
    assert result == ["result1", "result2"]


def test_get_miss():
    cache = ToolResultCache()
    assert cache.get("search", {"query": "missing"}) is None


def test_has_true():
    cache = ToolResultCache()
    cache.put("tool", {"x": 1}, "val")
    assert cache.has("tool", {"x": 1}) is True


def test_has_false():
    cache = ToolResultCache()
    assert cache.has("tool", {"x": 1}) is False


def test_invalidate():
    cache = ToolResultCache()
    cache.put("tool", {"x": 1}, "val")
    assert cache.invalidate("tool", {"x": 1}) is True
    assert cache.get("tool", {"x": 1}) is None


def test_invalidate_missing():
    cache = ToolResultCache()
    assert cache.invalidate("tool", {"x": 1}) is False


def test_ttl_expiry():
    cache = ToolResultCache(default_ttl=0.05)
    cache.put("tool", {"q": "a"}, "result")
    time.sleep(0.1)
    assert cache.get("tool", {"q": "a"}) is None


def test_ttl_not_expired():
    cache = ToolResultCache(default_ttl=60.0)
    cache.put("tool", {"q": "a"}, "result")
    assert cache.get("tool", {"q": "a"}) == "result"


def test_per_entry_ttl():
    cache = ToolResultCache(default_ttl=60.0)
    cache.put("tool", {"q": "a"}, "result", ttl=0.05)
    time.sleep(0.1)
    assert cache.get("tool", {"q": "a"}) is None


def test_lru_eviction():
    cache = ToolResultCache(max_size=2)
    cache.put("t", {"k": 1}, "r1")
    cache.put("t", {"k": 2}, "r2")
    cache.put("t", {"k": 3}, "r3")  # evicts k=1
    assert cache.size == 2
    assert cache.get("t", {"k": 1}) is None


def test_lru_access_order():
    cache = ToolResultCache(max_size=2)
    cache.put("t", {"k": 1}, "r1")
    cache.put("t", {"k": 2}, "r2")
    cache.get("t", {"k": 1})        # access k1 → MRU
    cache.put("t", {"k": 3}, "r3") # evicts k2 (LRU)
    assert cache.get("t", {"k": 2}) is None
    assert cache.get("t", {"k": 1}) == "r1"


def test_size():
    cache = ToolResultCache()
    cache.put("a", {}, 1)
    cache.put("b", {}, 2)
    assert cache.size == 2


def test_clear():
    cache = ToolResultCache()
    cache.put("t", {}, "v")
    cache.clear()
    assert cache.size == 0


def test_hits_misses():
    cache = ToolResultCache()
    cache.get("t", {"x": 1})  # miss
    cache.put("t", {"x": 1}, "v")
    cache.get("t", {"x": 1})  # hit
    assert cache.hits == 1
    assert cache.misses == 1


def test_hit_rate():
    cache = ToolResultCache()
    cache.put("t", {"x": 1}, "v")
    cache.get("t", {"x": 1})   # hit
    cache.get("t", {"x": 2})   # miss
    assert cache.hit_rate == pytest.approx(0.5)


def test_hit_rate_none_when_empty():
    cache = ToolResultCache()
    assert cache.hit_rate is None


def test_prune_expired():
    cache = ToolResultCache(default_ttl=0.05)
    cache.put("t", {"k": 1}, "r1")
    cache.put("t", {"k": 2}, "r2", ttl=60.0)
    time.sleep(0.1)
    removed = cache.prune_expired()
    assert removed == 1
    assert cache.get("t", {"k": 2}) == "r2"


def test_wrap_decorator():
    cache = ToolResultCache()
    calls = []

    @cache.wrap("my_tool")
    def my_tool(query: str) -> str:
        calls.append(query)
        return f"result:{query}"

    r1 = my_tool(query="python")
    r2 = my_tool(query="python")
    assert r1 == r2
    assert len(calls) == 1


def test_stats():
    cache = ToolResultCache()
    cache.put("t", {}, "v")
    cache.get("t", {})
    s = cache.stats
    assert s["hits"] == 1
    assert s["size"] == 1
