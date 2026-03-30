from __future__ import annotations

from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

# Keep class-level caches bounded to avoid unbounded memory growth.
DEFAULT_CACHE_MAX_SIZE = 32


class LRUDict(OrderedDict):
    """OrderedDict-based LRU cache with a fixed maximum size."""

    def __init__(self, maxsize: int = DEFAULT_CACHE_MAX_SIZE) -> None:
        super().__init__()
        self._maxsize = max(1, int(maxsize))

    def __setitem__(self, key: object, value: object) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self._maxsize:
            oldest_key, _ = next(iter(self.items()))
            del self[oldest_key]
            logger.debug("LRU cache eviction: removed oldest entry (cache size limit %d)", self._maxsize)

    def __getitem__(self, key: object) -> object:
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value
