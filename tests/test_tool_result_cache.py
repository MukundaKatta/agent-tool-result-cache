"""Unit tests for :mod:`agent_tool_result_cache`.

These tests use only the Python standard library (``unittest``) so they run
with no third-party dependencies::

    python3 -m unittest discover -s tests
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

# Allow running the suite directly against the source tree without an install.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_tool_result_cache import CacheEntry, ToolResultCache, __version__


class PutAndGetTests(unittest.TestCase):
    def test_put_then_get_returns_value(self) -> None:
        cache = ToolResultCache()
        cache.put("search", {"query": "python"}, ["r1", "r2"])
        self.assertEqual(cache.get("search", {"query": "python"}), ["r1", "r2"])

    def test_get_miss_returns_none(self) -> None:
        cache = ToolResultCache()
        self.assertIsNone(cache.get("search", {"query": "missing"}))

    def test_put_returns_key(self) -> None:
        cache = ToolResultCache()
        key = cache.put("t", {"x": 1}, "v")
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 64)  # sha256 hex digest

    def test_arg_order_does_not_change_key(self) -> None:
        cache = ToolResultCache()
        cache.put("t", {"a": 1, "b": 2}, "v")
        # Same args, different insertion order -> same logical key.
        self.assertEqual(cache.get("t", {"b": 2, "a": 1}), "v")

    def test_different_tools_do_not_collide(self) -> None:
        cache = ToolResultCache()
        cache.put("tool_a", {"x": 1}, "a")
        cache.put("tool_b", {"x": 1}, "b")
        self.assertEqual(cache.get("tool_a", {"x": 1}), "a")
        self.assertEqual(cache.get("tool_b", {"x": 1}), "b")

    def test_overwrite_updates_value(self) -> None:
        cache = ToolResultCache()
        cache.put("t", {"x": 1}, "first")
        cache.put("t", {"x": 1}, "second")
        self.assertEqual(cache.get("t", {"x": 1}), "second")
        self.assertEqual(cache.size, 1)

    def test_non_json_native_args_are_hashable(self) -> None:
        # default=str in the hasher keeps unusual values from raising.
        cache = ToolResultCache()
        cache.put("t", {"when": time.gmtime(0)}, "v")
        self.assertEqual(cache.get("t", {"when": time.gmtime(0)}), "v")


class HasTests(unittest.TestCase):
    def test_has_true_when_present(self) -> None:
        cache = ToolResultCache()
        cache.put("tool", {"x": 1}, "val")
        self.assertTrue(cache.has("tool", {"x": 1}))

    def test_has_false_when_absent(self) -> None:
        cache = ToolResultCache()
        self.assertFalse(cache.has("tool", {"x": 1}))

    def test_has_does_not_change_stats(self) -> None:
        cache = ToolResultCache()
        cache.put("tool", {"x": 1}, "val")
        cache.has("tool", {"x": 1})
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)

    def test_can_cache_a_none_value(self) -> None:
        cache = ToolResultCache()
        cache.put("tool", {"x": 1}, None)
        # get() cannot distinguish stored-None from a miss, but has() can.
        self.assertTrue(cache.has("tool", {"x": 1}))
        self.assertIsNone(cache.get("tool", {"x": 1}))


class InvalidateTests(unittest.TestCase):
    def test_invalidate_existing_returns_true(self) -> None:
        cache = ToolResultCache()
        cache.put("tool", {"x": 1}, "val")
        self.assertTrue(cache.invalidate("tool", {"x": 1}))
        self.assertIsNone(cache.get("tool", {"x": 1}))

    def test_invalidate_missing_returns_false(self) -> None:
        cache = ToolResultCache()
        self.assertFalse(cache.invalidate("tool", {"x": 1}))

    def test_invalidate_tool_removes_only_that_tool(self) -> None:
        cache = ToolResultCache()
        cache.put("search", {"q": "a"}, 1)
        cache.put("search", {"q": "b"}, 2)
        cache.put("other", {"q": "c"}, 3)
        removed = cache.invalidate_tool("search")
        self.assertEqual(removed, 2)
        self.assertIsNone(cache.get("search", {"q": "a"}))
        self.assertIsNone(cache.get("search", {"q": "b"}))
        self.assertEqual(cache.get("other", {"q": "c"}), 3)

    def test_invalidate_tool_unknown_returns_zero(self) -> None:
        cache = ToolResultCache()
        cache.put("search", {"q": "a"}, 1)
        self.assertEqual(cache.invalidate_tool("nope"), 0)
        self.assertEqual(cache.size, 1)


class TTLTests(unittest.TestCase):
    def test_default_ttl_expires(self) -> None:
        cache = ToolResultCache(default_ttl=0.05)
        cache.put("tool", {"q": "a"}, "result")
        time.sleep(0.1)
        self.assertIsNone(cache.get("tool", {"q": "a"}))

    def test_default_ttl_not_yet_expired(self) -> None:
        cache = ToolResultCache(default_ttl=60.0)
        cache.put("tool", {"q": "a"}, "result")
        self.assertEqual(cache.get("tool", {"q": "a"}), "result")

    def test_per_entry_ttl_overrides_default(self) -> None:
        cache = ToolResultCache(default_ttl=60.0)
        cache.put("tool", {"q": "a"}, "result", ttl=0.05)
        time.sleep(0.1)
        self.assertIsNone(cache.get("tool", {"q": "a"}))

    def test_none_ttl_never_expires(self) -> None:
        cache = ToolResultCache(default_ttl=None)
        cache.put("tool", {"q": "a"}, "result")
        self.assertFalse(CacheEntry(key="k", result="r").expired)
        self.assertEqual(cache.get("tool", {"q": "a"}), "result")

    def test_expired_get_counts_as_miss(self) -> None:
        cache = ToolResultCache(default_ttl=0.05)
        cache.put("tool", {"q": "a"}, "result")
        time.sleep(0.1)
        cache.get("tool", {"q": "a"})
        self.assertEqual(cache.misses, 1)


class LRUTests(unittest.TestCase):
    def test_eviction_drops_oldest(self) -> None:
        cache = ToolResultCache(max_size=2)
        cache.put("t", {"k": 1}, "r1")
        cache.put("t", {"k": 2}, "r2")
        cache.put("t", {"k": 3}, "r3")  # evicts k=1
        self.assertEqual(cache.size, 2)
        self.assertIsNone(cache.get("t", {"k": 1}))

    def test_get_refreshes_recency(self) -> None:
        cache = ToolResultCache(max_size=2)
        cache.put("t", {"k": 1}, "r1")
        cache.put("t", {"k": 2}, "r2")
        cache.get("t", {"k": 1})        # k1 becomes most-recently-used
        cache.put("t", {"k": 3}, "r3")  # evicts k2 (least recent)
        self.assertIsNone(cache.get("t", {"k": 2}))
        self.assertEqual(cache.get("t", {"k": 1}), "r1")

    def test_overwrite_does_not_evict(self) -> None:
        # Re-putting an existing key must not push the store over max_size and
        # spuriously evict another entry.
        cache = ToolResultCache(max_size=2)
        cache.put("t", {"k": 1}, "r1")
        cache.put("t", {"k": 2}, "r2")
        cache.put("t", {"k": 2}, "r2-updated")  # overwrite, not a new key
        self.assertEqual(cache.size, 2)
        self.assertEqual(cache.get("t", {"k": 1}), "r1")
        self.assertEqual(cache.get("t", {"k": 2}), "r2-updated")

    def test_max_size_one(self) -> None:
        cache = ToolResultCache(max_size=1)
        cache.put("t", {"k": 1}, "r1")
        cache.put("t", {"k": 2}, "r2")
        self.assertEqual(cache.size, 1)
        self.assertEqual(cache.get("t", {"k": 2}), "r2")

    def test_invalid_max_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            ToolResultCache(max_size=0)


class MaintenanceTests(unittest.TestCase):
    def test_size_counts_entries(self) -> None:
        cache = ToolResultCache()
        cache.put("a", {}, 1)
        cache.put("b", {}, 2)
        self.assertEqual(cache.size, 2)

    def test_clear_empties_and_resets_stats(self) -> None:
        cache = ToolResultCache()
        cache.put("t", {}, "v")
        cache.get("t", {})
        cache.clear()
        self.assertEqual(cache.size, 0)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)

    def test_prune_expired_only_removes_expired(self) -> None:
        cache = ToolResultCache(default_ttl=0.05)
        cache.put("t", {"k": 1}, "r1")
        cache.put("t", {"k": 2}, "r2", ttl=60.0)
        time.sleep(0.1)
        self.assertEqual(cache.prune_expired(), 1)
        self.assertEqual(cache.get("t", {"k": 2}), "r2")


class StatsTests(unittest.TestCase):
    def test_hits_and_misses(self) -> None:
        cache = ToolResultCache()
        cache.get("t", {"x": 1})  # miss
        cache.put("t", {"x": 1}, "v")
        cache.get("t", {"x": 1})  # hit
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

    def test_hit_rate(self) -> None:
        cache = ToolResultCache()
        cache.put("t", {"x": 1}, "v")
        cache.get("t", {"x": 1})  # hit
        cache.get("t", {"x": 2})  # miss
        self.assertAlmostEqual(cache.hit_rate, 0.5)

    def test_hit_rate_none_when_empty(self) -> None:
        cache = ToolResultCache()
        self.assertIsNone(cache.hit_rate)

    def test_stats_snapshot(self) -> None:
        cache = ToolResultCache()
        cache.put("t", {}, "v")
        cache.get("t", {})
        snapshot = cache.stats
        self.assertEqual(snapshot["hits"], 1)
        self.assertEqual(snapshot["size"], 1)
        self.assertIn("misses", snapshot)
        self.assertIn("hit_rate", snapshot)


class WrapTests(unittest.TestCase):
    def test_wrap_caches_after_first_call(self) -> None:
        cache = ToolResultCache()
        calls: list[str] = []

        @cache.wrap("my_tool")
        def my_tool(query: str) -> str:
            calls.append(query)
            return f"result:{query}"

        r1 = my_tool(query="python")
        r2 = my_tool(query="python")
        self.assertEqual(r1, r2)
        self.assertEqual(len(calls), 1)

    def test_wrap_preserves_function_metadata(self) -> None:
        cache = ToolResultCache()

        @cache.wrap("my_tool")
        def my_tool(query: str) -> str:
            """Look something up."""
            return query

        self.assertEqual(my_tool.__name__, "my_tool")
        self.assertEqual(my_tool.__doc__, "Look something up.")

    def test_wrap_caches_none_result(self) -> None:
        # A wrapped function that returns None must still be cached so it is
        # only executed once for the same arguments.
        cache = ToolResultCache()
        calls: list[int] = []

        @cache.wrap("returns_none")
        def returns_none(x: int) -> None:
            calls.append(x)
            return None

        self.assertIsNone(returns_none(x=1))
        self.assertIsNone(returns_none(x=1))
        self.assertEqual(len(calls), 1)

    def test_wrap_distinguishes_kwargs(self) -> None:
        cache = ToolResultCache()

        @cache.wrap("echo")
        def echo(value: int) -> int:
            return value * 2

        self.assertEqual(echo(value=2), 4)
        self.assertEqual(echo(value=3), 6)


class PackageMetadataTests(unittest.TestCase):
    def test_version_string(self) -> None:
        self.assertIsInstance(__version__, str)
        self.assertTrue(__version__)


if __name__ == "__main__":
    unittest.main()
