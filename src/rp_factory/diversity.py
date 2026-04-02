"""多样性监控与种子去重追踪。

实现 inspire.md §1.3 的多样性保障：
  - n-gram 多样性检测：当生成文本的 n-gram 重复率超过阈值时发出警告
  - 种子去重追踪：同一批次内避免重复使用同一个种子
  - 种子使用统计：追踪每个种子被选中的次数，为动态扩充提供依据
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from typing import Any

logger = logging.getLogger(__name__)


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
            old_ngrams = self._extract_ngrams(old)
            for ng in old_ngrams:
                self._ngram_counts[ng] -= 1
                if self._ngram_counts[ng] <= 0:
                    del self._ngram_counts[ng]

        return self.get_collapse_ratio()

    def get_collapse_ratio(self) -> float:
        """计算当前窗口中出现超过 1 次的 n-gram 占比。"""
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


class SeedTracker:
    """种子去重追踪器 — 同一批次内避免重复选取。"""

    def __init__(self) -> None:
        self._used: dict[str, set[int]] = defaultdict(set)
        self._usage_counts: dict[str, Counter[int]] = defaultdict(Counter)

    def pick_unique(self, pool_name: str, items: list[Any], max_retries: int = 10) -> tuple[int, Any]:
        """从列表中选一个本批次尚未使用过的条目。

        Returns:
            (index, item) — 如果全部用过则重置追踪再选。
        """
        used = self._used[pool_name]
        available = [i for i in range(len(items)) if i not in used]

        if not available:
            used.clear()
            available = list(range(len(items)))

        idx = random.choice(available)
        used.add(idx)
        self._usage_counts[pool_name][idx] += 1
        return idx, items[idx]

    def get_stats(self, pool_name: str) -> dict[str, Any]:
        """返回指定种子池的使用统计。"""
        counts = self._usage_counts.get(pool_name, Counter())
        if not counts:
            return {"total_picks": 0, "unique_used": 0, "most_used": None}
        return {
            "total_picks": sum(counts.values()),
            "unique_used": len(counts),
            "most_used": counts.most_common(3),
            "least_used": counts.most_common()[-3:] if len(counts) >= 3 else counts.most_common(),
        }

    def reset_batch(self) -> None:
        """新批次开始时重置使用记录（保留统计）。"""
        for k in self._used:
            self._used[k].clear()

    def reset_all(self) -> None:
        self._used.clear()
        self._usage_counts.clear()


class DiversityGuard:
    """统一的多样性保障入口。"""

    def __init__(
        self,
        ngram_n: int = 3,
        window_size: int = 50,
        collapse_threshold: float = 0.7,
    ) -> None:
        self.intent_monitor = NGramMonitor(ngram_n, window_size, collapse_threshold)
        self.event_monitor = NGramMonitor(ngram_n, window_size, collapse_threshold)
        self.output_monitor = NGramMonitor(ngram_n, window_size, collapse_threshold)
        self.seed_tracker = SeedTracker()

    def check_and_warn(self, category: str, text: str) -> float:
        """添加文本到对应监控器，如果塌缩则发出警告。返回当前重复率。"""
        monitor_map = {
            "intent": self.intent_monitor,
            "event": self.event_monitor,
            "output": self.output_monitor,
        }
        monitor = monitor_map.get(category, self.output_monitor)
        ratio = monitor.add_text(text)
        if monitor.is_collapsing():
            logger.warning(
                "多样性警告 [%s]: n-gram 重复率 %.2f 超过阈值 %.2f，建议扩充种子库",
                category, ratio, monitor.threshold,
            )
        return ratio

    def reset(self) -> None:
        self.intent_monitor.reset()
        self.event_monitor.reset()
        self.output_monitor.reset()
        self.seed_tracker.reset_all()
