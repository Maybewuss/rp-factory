"""测试 Pool 类。"""

import pytest

from rp_factory.diversity import Pool


class TestPool:
    def test_pick_returns_item(self):
        pool = Pool(["A", "B", "C"])
        assert pool.pick() in ("A", "B", "C")

    def test_no_repeat_until_exhausted(self):
        pool = Pool(["A", "B", "C"])
        first_round = [pool.pick() for _ in range(3)]
        assert len(set(first_round)) == 3

    def test_reset_after_exhaustion(self):
        pool = Pool(["X", "Y"])
        pool.pick()
        pool.pick()
        result = pool.pick()
        assert result in ("X", "Y")

    def test_add_new_item(self):
        pool = Pool(["A"])
        assert pool.add("B") is True
        assert len(pool) == 2

    def test_add_duplicate(self):
        pool = Pool(["A"])
        assert pool.add("A") is False
        assert len(pool) == 1

    def test_extend(self):
        pool = Pool(["A"])
        added = pool.extend(["B", "C", "A"])
        assert added == 2
        assert len(pool) == 3

    def test_empty_pool_raises(self):
        pool = Pool()
        with pytest.raises(ValueError):
            pool.pick()

    def test_items_property(self):
        pool = Pool(["A", "B"])
        assert pool.items == ["A", "B"]
