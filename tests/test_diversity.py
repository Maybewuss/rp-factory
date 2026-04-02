"""测试 n-gram 多样性监控。"""

from rp_factory.diversity import NGramMonitor


class TestNGramMonitor:
    def test_no_collapse_diverse_texts(self):
        monitor = NGramMonitor(n=3, window_size=10, threshold=0.7)
        for t in ["今天天气真好啊", "昨天看了一部电影", "下周要去出差", "午饭吃了什么", "明天有个会议"]:
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
        for _ in range(3):
            monitor.add_text(same)
        r1 = monitor.get_collapse_ratio()
        for t in ["昨晚做了一个奇怪的梦", "周末打算去爬山放松", "刚买了一本新的小说"]:
            monitor.add_text(t)
        assert monitor.get_collapse_ratio() < r1
