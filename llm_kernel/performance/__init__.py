"""性能测试：给代码计时。

只依赖标准库。新增计时/性能工具的配方见 docs/guide/design.md。
"""

from . import timing
from .timing import Timer, timer

__all__ = ["Timer", "timer", "timing"]
