"""测试多样性监控与种子追踪。"""

from rp_factory.diversity import DiversityGuard, NGramMonitor, SeedTracker


class TestNGramMonitor:
    def test_no_collapse_diverse_texts(self):
        monitor = NGramMonitor(n=3, window_size=10, threshold=0.7)
        texts = [
            "今天天气真好啊",
            "昨天看了一部电影",
            "下周要去出差",
            "午饭吃了什么",
            "明天有个会议",
        ]
        for t in texts:
            monitor.add_text(t)
        assert not monitor.is_collapsing()

    def test_detects_collapse_repeated_texts(self):
        monitor = NGramMonitor(n=3, window_size=10, threshold=0.5)
        for _ in range(10):
            monitor.add_text("我真是个纯废物连杯咖啡都拿不稳")
        assert monitor.is_collapsing()

    def test_reset_clears_state(self):
        monitor = NGramMonitor(n=3, window_size=5, threshold=0.5)
        for _ in range(5):
            monitor.add_text("重复内容重复内容")
        assert monitor.is_collapsing()
        monitor.reset()
        assert not monitor.is_collapsing()

    def test_window_sliding(self):
        monitor = NGramMonitor(n=2, window_size=3, threshold=0.9)
        same = "今天天气真好出门走走"
        monitor.add_text(same)
        monitor.add_text(same)
        monitor.add_text(same)
        r1 = monitor.get_collapse_ratio()
        monitor.add_text("昨晚做了一个奇怪的梦")
        monitor.add_text("周末打算去爬山放松")
        monitor.add_text("刚买了一本新的小说")
        r2 = monitor.get_collapse_ratio()
        assert r2 < r1


class TestSeedTracker:
    def test_no_repeat_within_pool(self):
        tracker = SeedTracker()
        items = ["A", "B", "C", "D", "E"]
        picked = set()
        for _ in range(5):
            _, item = tracker.pick_unique("test", items)
            picked.add(item)
        assert picked == {"A", "B", "C", "D", "E"}

    def test_resets_after_exhaustion(self):
        tracker = SeedTracker()
        items = ["X", "Y"]
        tracker.pick_unique("p", items)
        tracker.pick_unique("p", items)
        _, third = tracker.pick_unique("p", items)
        assert third in ("X", "Y")

    def test_stats_tracking(self):
        tracker = SeedTracker()
        items = ["A", "B", "C"]
        for _ in range(6):
            tracker.pick_unique("s", items)
        stats = tracker.get_stats("s")
        assert stats["total_picks"] == 6
        assert stats["unique_used"] == 3

    def test_batch_reset_preserves_stats(self):
        tracker = SeedTracker()
        items = ["A", "B"]
        tracker.pick_unique("q", items)
        tracker.reset_batch()
        stats = tracker.get_stats("q")
        assert stats["total_picks"] == 1

    def test_independent_pools(self):
        tracker = SeedTracker()
        tracker.pick_unique("pool1", ["A"])
        tracker.pick_unique("pool2", ["B"])
        s1 = tracker.get_stats("pool1")
        s2 = tracker.get_stats("pool2")
        assert s1["total_picks"] == 1
        assert s2["total_picks"] == 1


class TestDiversityGuard:
    def test_check_and_warn_returns_ratio(self):
        guard = DiversityGuard()
        ratio = guard.check_and_warn("intent", "测试文本一")
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0

    def test_reset_clears_all(self):
        guard = DiversityGuard()
        for _ in range(5):
            guard.check_and_warn("output", "同样的话同样的话")
        guard.reset()
        assert not guard.output_monitor.is_collapsing()
