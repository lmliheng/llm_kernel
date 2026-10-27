"""计时工具（只用标准库）。

和 0.1.x 相比的改动：改用单调时钟 :func:`time.perf_counter`、``stop()`` 前没 ``start()``
会给明确的报错、支持 ``with`` 写法、``avg()`` 在没记录时报错而不是 ``ZeroDivisionError``。
"""

from __future__ import annotations

import time as _time
from functools import wraps
from itertools import accumulate
from typing import Callable

__all__ = ["Timer", "timer"]


class Timer:
    """给一段代码计时，可以记多次。

    用法一（手动）：::

        t = Timer()
        t.start()
        ...
        t.stop()

    用法二（``with``，自动记一笔）：::

        with Timer() as t:
            ...

    Examples:
        >>> import time
        >>> from llm_kernel.performance import Timer
        >>> t = Timer()
        >>> t.start(); _ = time.sleep(0.001); elapsed = t.stop()
        >>> len(t.times), t.sum() == elapsed, round(t.avg(), 12) == round(elapsed, 12)
        (1, True, True)
    """

    def __init__(self, *, auto_start: bool = False) -> None:
        self.times: list = []
        self.tik = None
        if auto_start:
            self.start()

    def start(self):
        """开始计时（重复调用会覆盖上一次的起点）。"""
        self.tik = _time.perf_counter()

    def stop(self) -> float:
        """停止计时，把这次耗时记进 :attr:`times` 并返回它（秒）。"""
        if self.tik is None:
            raise RuntimeError("还没有 start()，不能 stop()；请先调用 t.start() 或用 with Timer() as t")
        elapsed = _time.perf_counter() - self.tik
        self.tik = None
        self.times.append(elapsed)
        return elapsed

    @property
    def last(self) -> float:
        """最近一次的耗时。"""
        if not self.times:
            raise ValueError("还没有任何计时结果，先 start()/stop() 或者用 with Timer() as t")
        return self.times[-1]

    @property
    def count(self) -> int:
        """已经记了几次。"""
        return len(self.times)

    @property
    def elapsed(self) -> float:
        """从 ``start()`` 到现在过了多久（不停止计时）。"""
        if self.tik is None:
            raise RuntimeError("计时器没有在运行，请先调用 start()")
        return _time.perf_counter() - self.tik

    def sum(self) -> float:
        """所有记录的总耗时（秒）。"""
        return float(sum(self.times))

    def avg(self) -> float:
        """平均耗时（秒）。"""
        if not self.times:
            raise ValueError("还没有任何计时结果，先 start()/stop() 或者用 with Timer() as t")
        return self.sum() / len(self.times)

    def cumsum(self) -> list:
        """累计耗时序列，例如 ``[0.1, 0.3, 0.6]``（原来依赖 numpy，现在只用标准库）。"""
        return list(accumulate(self.times))

    def reset(self) -> None:
        """清空所有记录。"""
        self.times = []
        self.tik = None

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *exc_info) -> bool:
        self.stop()
        return False

    def __repr__(self) -> str:
        return f"<Timer 记录了 {self.count} 次，合计 {self.sum():.6f}s>"


def timer(
    func: Callable | None = None,
    *,
    verbose: bool = True,
    precision: int = 4,
) -> Callable:
    """打印函数耗时的装饰器，也可以用 ``@timer(verbose=False)`` 静默计时。

    计时结果同时挂在 ``wrapper.times`` / ``wrapper.last`` 上，异常也会被计时（用 ``finally``）。

    Args:
        func (Callable | None): 被装饰的函数；``@timer`` 直接用时由 Python 传入。
        verbose (bool): 是否 ``print`` 一行耗时。
        precision (int): 秒数保留几位小数。

    Returns:
        Callable: 包装后的函数，返回值与被装饰函数一致。

    Examples:
        >>> import time
        >>> from llm_kernel.performance import timer
        >>> @timer(verbose=False)
        ... def work(n):
        ...     time.sleep(0.001)
        ...     return n * 2
        >>> work(21)
        42
        >>> len(work.times)
        1
    """

    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = _time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = _time.perf_counter() - start
                wrapper.times.append(elapsed)
                wrapper.last = elapsed
                if verbose:
                    print(f"{fn.__name__} 耗时: {elapsed:.{precision}f}秒")

        wrapper.times = []
        wrapper.last = None
        return wrapper

    if func is None:  # 带参数的用法：@timer(verbose=False)
        return decorate
    return decorate(func)
