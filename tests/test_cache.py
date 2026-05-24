import asyncio
import threading
import time

import pytest

from agent_tool_result_cache import ToolResultCache

# ---------- basic set/get ----------


def test_set_then_get_round_trip():
    c = ToolResultCache()
    c.set("search", {"query": "foo"}, result=["a", "b"])
    assert c.get("search", {"query": "foo"}) == ["a", "b"]


def test_get_miss_returns_none():
    c = ToolResultCache()
    assert c.get("search", {"query": "never-set"}) is None


def test_get_after_set_different_args_misses():
    c = ToolResultCache()
    c.set("search", {"query": "foo"}, result=1)
    assert c.get("search", {"query": "bar"}) is None
    assert c.get("search", {"query": "foo"}) == 1


def test_contains_does_not_bump_lru_or_stats():
    c = ToolResultCache(maxsize=2)
    c.set("t", {"i": 1}, result="a")
    c.set("t", {"i": 2}, result="b")
    # contains() should not promote i=1
    assert c.contains("t", {"i": 1}) is True
    # add a 3rd -> the LRU element should be i=1
    c.set("t", {"i": 3}, result="c")
    assert c.get("t", {"i": 1}) is None  # evicted
    assert c.get("t", {"i": 2}) == "b"
    assert c.get("t", {"i": 3}) == "c"
    # contains did not move stats: only the 3 get calls above and any internal lookups
    stats = c.stats()
    assert stats["hits"] == 2  # the two gets that found
    # one miss from i=1 lookup
    assert stats["misses"] >= 1


def test_args_order_independent_canonical_hash():
    c = ToolResultCache()
    c.set("t", {"a": 1, "b": 2}, result="x")
    assert c.get("t", {"b": 2, "a": 1}) == "x"


# ---------- TTL ----------


def test_ttl_expires(monkeypatch):
    c = ToolResultCache(default_ttl_seconds=5)
    now = [1000.0]

    def fake_monotonic():
        return now[0]

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    c.set("t", {"k": 1}, result="v")
    assert c.get("t", {"k": 1}) == "v"
    now[0] += 6
    assert c.get("t", {"k": 1}) is None  # expired


def test_per_tool_ttl_override(monkeypatch):
    c = ToolResultCache(default_ttl_seconds=600)
    c.set_ttl("short_tool", ttl_seconds=1)
    c.set_ttl("long_tool", ttl_seconds=10_000)

    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    c.set("short_tool", {"x": 1}, result="s")
    c.set("long_tool", {"x": 1}, result="l")

    now[0] += 5
    assert c.get("short_tool", {"x": 1}) is None  # expired
    assert c.get("long_tool", {"x": 1}) == "l"  # still valid


def test_ttl_zero_never_caches():
    c = ToolResultCache()
    c.set_ttl("get_current_time", ttl_seconds=0)
    c.set("get_current_time", {}, result="2026-05-24T10:00:00Z")
    # immediate get should miss
    assert c.get("get_current_time", {}) is None
    assert c.size == 0


def test_explicit_ttl_arg_overrides_default():
    c = ToolResultCache(default_ttl_seconds=600)
    c.set("t", {"k": 1}, result="v", ttl_seconds=0)  # opt-out this one row
    assert c.get("t", {"k": 1}) is None


def test_invalid_ttl_raises():
    with pytest.raises(ValueError):
        ToolResultCache(default_ttl_seconds=-1)
    c = ToolResultCache()
    with pytest.raises(ValueError):
        c.set_ttl("t", ttl_seconds=-5)


def test_invalid_maxsize_raises():
    with pytest.raises(ValueError):
        ToolResultCache(maxsize=0)
    with pytest.raises(ValueError):
        ToolResultCache(maxsize=-1)


# ---------- LRU ----------


def test_lru_eviction_at_maxsize():
    c = ToolResultCache(maxsize=3)
    c.set("t", {"i": 1}, result="a")
    c.set("t", {"i": 2}, result="b")
    c.set("t", {"i": 3}, result="c")
    c.set("t", {"i": 4}, result="d")
    # i=1 should have been evicted
    assert c.get("t", {"i": 1}) is None
    assert c.get("t", {"i": 4}) == "d"
    assert c.size == 3


def test_lru_get_promotes_recency():
    c = ToolResultCache(maxsize=3)
    c.set("t", {"i": 1}, result="a")
    c.set("t", {"i": 2}, result="b")
    c.set("t", {"i": 3}, result="c")
    # touch i=1 -> now LRU is i=2
    assert c.get("t", {"i": 1}) == "a"
    c.set("t", {"i": 4}, result="d")
    assert c.get("t", {"i": 2}) is None  # evicted
    assert c.get("t", {"i": 1}) == "a"  # survived


def test_set_replaces_existing_key():
    c = ToolResultCache()
    c.set("t", {"k": 1}, result="v1")
    c.set("t", {"k": 1}, result="v2")
    assert c.get("t", {"k": 1}) == "v2"
    assert c.size == 1


# ---------- decorator ----------


def test_cached_decorator_works():
    c = ToolResultCache()
    calls = []

    @c.cached("search")
    def search(query: str) -> list:
        calls.append(query)
        return [query.upper()]

    assert search("hello") == ["HELLO"]
    assert search("hello") == ["HELLO"]
    assert calls == ["hello"]  # only one real call


def test_cached_decorator_caches_by_args():
    c = ToolResultCache()
    calls = []

    @c.cached("search")
    def search(query: str, limit: int = 10) -> str:
        calls.append((query, limit))
        return f"{query}/{limit}"

    search("x")  # default limit
    search("x")  # hit
    search("x", limit=5)  # different args, miss
    search("x", limit=5)  # hit
    assert calls == [("x", 10), ("x", 5)]


def test_decorator_skips_cache_when_ttl_zero():
    c = ToolResultCache()
    c.set_ttl("time_tool", ttl_seconds=0)
    calls = []

    @c.cached("time_tool")
    def now_str() -> str:
        calls.append(1)
        return "t"

    now_str()
    now_str()
    now_str()
    assert len(calls) == 3  # never cached


# ---------- stats ----------


def test_stats_per_tool_and_total():
    c = ToolResultCache()
    c.set("a", {"k": 1}, result="x")
    c.set("b", {"k": 1}, result="y")
    c.get("a", {"k": 1})  # hit a
    c.get("a", {"k": 1})  # hit a
    c.get("a", {"k": 999})  # miss a
    c.get("b", {"k": 1})  # hit b
    c.get("c", {"k": 1})  # miss c (tool not seen for set)

    s = c.stats()
    assert s["hits"] == 3
    assert s["misses"] == 2
    assert s["per_tool"]["a"]["hits"] == 2
    assert s["per_tool"]["a"]["misses"] == 1
    assert s["per_tool"]["b"]["hits"] == 1
    assert s["per_tool"]["b"]["misses"] == 0
    assert s["per_tool"]["c"]["misses"] == 1
    assert s["size"] == 2


def test_stats_track_evictions():
    c = ToolResultCache(maxsize=2)
    c.set("t", {"i": 1}, result="a")
    c.set("t", {"i": 2}, result="b")
    c.set("t", {"i": 3}, result="c")  # evicts i=1
    s = c.stats()
    assert s["evictions"] == 1
    assert s["per_tool"]["t"]["evictions"] == 1


def test_stats_track_expirations(monkeypatch):
    c = ToolResultCache(default_ttl_seconds=5)
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    c.set("t", {"k": 1}, result="v")
    now[0] += 10
    c.get("t", {"k": 1})  # expired
    s = c.stats()
    assert s["expirations"] == 1
    assert s["per_tool"]["t"]["expirations"] == 1


# ---------- delete / invalidate ----------


def test_invalidate_single_tool():
    c = ToolResultCache()
    c.set("a", {"k": 1}, result="x")
    c.set("a", {"k": 2}, result="y")
    c.set("b", {"k": 1}, result="z")
    removed = c.invalidate("a")
    assert removed == 2
    assert c.get("a", {"k": 1}) is None
    assert c.get("b", {"k": 1}) == "z"


def test_invalidate_all():
    c = ToolResultCache()
    c.set("a", {"k": 1}, result="x")
    c.set("b", {"k": 1}, result="y")
    removed = c.invalidate()
    assert removed == 2
    assert c.size == 0


def test_delete_one_entry():
    c = ToolResultCache()
    c.set("t", {"k": 1}, result="x")
    c.set("t", {"k": 2}, result="y")
    assert c.delete("t", {"k": 1}) == 1
    assert c.get("t", {"k": 1}) is None
    assert c.get("t", {"k": 2}) == "y"
    # second delete is a no-op
    assert c.delete("t", {"k": 1}) == 0


def test_delete_drops_all_for_tool_when_args_none():
    c = ToolResultCache()
    c.set("a", {"k": 1}, result="x")
    c.set("a", {"k": 2}, result="y")
    c.set("b", {"k": 1}, result="z")
    assert c.delete("a") == 2
    assert c.get("a", {"k": 1}) is None
    assert c.get("b", {"k": 1}) == "z"


# ---------- persistence ----------


def test_persistence_save_and_load_round_trip(tmp_path):
    p = tmp_path / "cache.jsonl"
    c1 = ToolResultCache(persist_path=p)
    c1.set("search", {"q": "foo"}, result=["a", "b"])
    c1.set("weather", {"city": "Dallas"}, result={"temp": 80})
    assert p.exists()

    c2 = ToolResultCache(persist_path=p)
    assert c2.get("search", {"q": "foo"}) == ["a", "b"]
    assert c2.get("weather", {"city": "Dallas"}) == {"temp": 80}


def test_persistence_skips_expired_on_load(tmp_path):
    p = tmp_path / "cache.jsonl"
    c1 = ToolResultCache(default_ttl_seconds=1, persist_path=p)
    c1.set("t", {"k": "fresh"}, result="v", ttl_seconds=3600)
    c1.set("t", {"k": "stale"}, result="v")  # 1s TTL
    time.sleep(1.2)
    c2 = ToolResultCache(persist_path=p)
    assert c2.get("t", {"k": "fresh"}) == "v"
    assert c2.get("t", {"k": "stale"}) is None


def test_persistence_latest_write_wins(tmp_path):
    p = tmp_path / "cache.jsonl"
    c1 = ToolResultCache(persist_path=p)
    c1.set("t", {"k": 1}, result="old")
    c1.set("t", {"k": 1}, result="new")
    c2 = ToolResultCache(persist_path=p)
    assert c2.get("t", {"k": 1}) == "new"


def test_no_persistence_when_path_none(tmp_path):
    c = ToolResultCache()
    c.set("t", {"k": 1}, result="x")
    # nothing was written anywhere
    assert not list(tmp_path.iterdir())


# ---------- async ----------


def test_cached_async_works():
    c = ToolResultCache()
    calls = []

    @c.cached_async("fetch_url")
    async def fetch_url(url: str) -> str:
        calls.append(url)
        await asyncio.sleep(0)
        return f"body-of-{url}"

    async def run():
        a = await fetch_url("https://example.com")
        b = await fetch_url("https://example.com")
        return a, b

    a, b = asyncio.run(run())
    assert a == "body-of-https://example.com"
    assert b == a
    assert calls == ["https://example.com"]


def test_cached_async_distinct_args():
    c = ToolResultCache()
    calls = []

    @c.cached_async("fetch_url")
    async def fetch_url(url: str) -> str:
        calls.append(url)
        return url[::-1]

    async def run():
        await fetch_url("a")
        await fetch_url("b")
        await fetch_url("a")
        await fetch_url("b")

    asyncio.run(run())
    assert sorted(calls) == ["a", "b"]


# ---------- thread safety ----------


def test_concurrent_set_and_get():
    c = ToolResultCache(maxsize=10_000)
    errors = []

    def writer(start: int) -> None:
        try:
            for i in range(start, start + 100):
                c.set("t", {"i": i}, result=i * 2)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader(start: int) -> None:
        try:
            for i in range(start, start + 100):
                # value may or may not be present yet; both are fine
                v = c.get("t", {"i": i})
                assert v is None or v == i * 2
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = []
    for k in range(10):
        threads.append(threading.Thread(target=writer, args=(k * 100,)))
        threads.append(threading.Thread(target=reader, args=(k * 100,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # all 1000 entries should ultimately be readable
    for i in range(1000):
        assert c.get("t", {"i": i}) == i * 2


def test_concurrent_eviction_does_not_break_invariant():
    """maxsize must hold even under concurrent set traffic."""
    c = ToolResultCache(maxsize=50)

    def writer(start: int) -> None:
        for i in range(start, start + 50):
            c.set("t", {"i": i}, result=i)

    threads = [threading.Thread(target=writer, args=(k * 50,)) for k in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.size == 50
