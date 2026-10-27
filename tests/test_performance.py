"""`llm_kernel.performance` 的测试，覆盖 0.1.x 里 `avg()` 除零等问题。"""

from __future__ import annotations

import time

import pytest

from llm_kernel.performance import Timer, timer


def _burn(seconds: float = 0.001) -> None:
    time.sleep(seconds)


def test_records_and_aggregates():
    t = Timer()
    t.start()
    _burn()
    first = t.stop()
    t.start()
    _burn()
    second = t.stop()

    assert t.times == [first, second]
    assert t.count == 2
    assert t.last == second
    assert t.sum() == pytest.approx(first + second)
    assert t.avg() == pytest.approx((first + second) / 2)
    assert t.cumsum() == pytest.approx([first, first + second])


def test_context_manager_records_one_sample():
    with Timer() as t:
        _burn()
    assert t.count == 1
    assert t.last > 0


def test_measures_real_elapsed_time():
    t = Timer(auto_start=True)
    _burn(0.02)
    assert t.stop() >= 0.02


def test_reset():
    with Timer() as t:
        _burn()
    t.reset()
    assert t.times == []


def test_clear_errors_instead_of_zero_division():
    with pytest.raises(RuntimeError, match="还没有 start"):
        Timer().stop()
    with pytest.raises(ValueError, match="还没有任何计时结果"):
        Timer().avg()
    with pytest.raises(ValueError, match="还没有任何计时结果"):
        Timer().last
    with pytest.raises(RuntimeError, match="没有在运行"):
        Timer().elapsed


def test_elapsed_while_running():
    t = Timer(auto_start=True)
    _burn()
    assert t.elapsed > 0


def test_sum_and_cumsum_are_plain_floats():
    with Timer() as t:
        _burn()
    assert isinstance(t.sum(), float)
    assert isinstance(t.cumsum(), list)


def test_decorator_prints_and_returns_the_value(capsys):
    @timer
    def add(a, b):
        _burn()
        return a + b

    assert add(1, 2) == 3
    assert "add 耗时" in capsys.readouterr().out
    assert add.times and add.last is not None


def test_decorator_can_be_silent(capsys):
    @timer(verbose=False)
    def work():
        _burn()
        return "done"

    assert work() == "done"
    assert capsys.readouterr().out == ""
    assert len(work.times) == 1


def test_decorator_times_exceptions_too():
    @timer(verbose=False)
    def boom():
        _burn()
        raise RuntimeError("故意失败")

    with pytest.raises(RuntimeError):
        boom()
    assert len(boom.times) == 1
