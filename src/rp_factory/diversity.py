"""多样性工具：无重复池 + n-gram 监控。"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Pool:
    """通用无重复选取池。用完一轮自动重置。

    统一替代旧的 PromptPool 和 SeedTracker。
    """

    def __init__(self, items: list[Any] | None = None) -> None:
        self._items: list[Any] = list(items or [])
        self._used: set[int] = set()

    def pick(self) -> Any:
        if not self._items:
            raise ValueError("Pool is empty")
        available = [i for i in range(len(self._items)) if i not in self._used]
        if not available:
            self._used.clear()
            available = list(range(len(self._items)))
        idx = random.choice(available)
        self._used.add(idx)
        return self._items[idx]

    def add(self, item: Any) -> bool:
        """添加新条目。如果已存在则跳过，返回是否实际添加。"""
        if item in self._items:
            return False
        self._items.append(item)
        return True

    def extend(self, items: list[Any]) -> int:
        """批量添加，返回实际新增数量。"""
        return sum(1 for item in items if self.add(item))

    def reset(self) -> None:
        self._used.clear()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> list[Any]:
        return list(self._items)


class NGramMonitor:
    """滑动窗口 n-gram 多样性监控器。"""

    def __init__(self, n: int = 3, window_size: int = 50, threshold: float = 0.7) -> None:
        self.n = n
        self.window_size = window_size
        self.threshold = threshold
        self._recent_texts: list[str] = []
        self._ngram_counts: Counter[tuple[str, ...]] = Counter()

    def _extract_ngrams(self, text: str) -> list[tuple[str, ...]]:
        chars = list(text.replace(" ", "").replace("\n", ""))
        if len(chars) < self.n:
            return [tuple(chars)]
        return [tuple(chars[i:i + self.n]) for i in range(len(chars) - self.n + 1)]

    def add_text(self, text: str) -> float:
        """添加一条文本并返回当前窗口的重复率 (0~1)。"""
        ngrams = self._extract_ngrams(text)
        for ng in ngrams:
            self._ngram_counts[ng] += 1

        self._recent_texts.append(text)

        if len(self._recent_texts) > self.window_size:
            old = self._recent_texts.pop(0)
            for ng in self._extract_ngrams(old):
                self._ngram_counts[ng] -= 1
                if self._ngram_counts[ng] <= 0:
                    del self._ngram_counts[ng]

        return self.get_collapse_ratio()

    def get_collapse_ratio(self) -> float:
        if not self._ngram_counts:
            return 0.0
        total = sum(self._ngram_counts.values())
        repeated = sum(v for v in self._ngram_counts.values() if v > 1)
        return repeated / total if total > 0 else 0.0

    def is_collapsing(self) -> bool:
        return self.get_collapse_ratio() > self.threshold

    def reset(self) -> None:
        self._recent_texts.clear()
        self._ngram_counts.clear()
