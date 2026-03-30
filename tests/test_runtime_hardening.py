from __future__ import annotations

import unittest

from kerrtrace.cache_utils import LRUDict
from kerrtrace.raytracer import _LRUDict


class RuntimeHardeningTests(unittest.TestCase):
    def test_lru_dict_evicts_oldest(self) -> None:
        cache = LRUDict(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        self.assertNotIn("a", cache)
        self.assertIn("b", cache)
        self.assertIn("c", cache)

    def test_lru_dict_updates_recency_on_get(self) -> None:
        cache = LRUDict(maxsize=2)
        cache["a"] = 1
        cache["b"] = 2
        _ = cache["a"]  # refresh recency of "a"
        cache["c"] = 3  # should evict "b"
        self.assertIn("a", cache)
        self.assertIn("c", cache)
        self.assertNotIn("b", cache)

    def test_raytracer_alias_kept_for_backwards_compatibility(self) -> None:
        self.assertIs(_LRUDict, LRUDict)


if __name__ == "__main__":
    unittest.main()
