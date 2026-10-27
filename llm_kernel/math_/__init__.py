"""兼容层：``llm_kernel.math_`` 已更名为 :mod:`llm_kernel.math`。

0.1.x 里这个目录叫 ``math_``（当年为了不和标准库 ``math`` 撞名）；现在名字统一收进
``llm_kernel`` 包内，下划线不再需要。这里只做转发 + 提醒一次，1.0 版本删除。
"""

import warnings

from .. import math as _current
from ..math import *  # noqa: F401,F403  —— 把新模块的公开名字原样搬过来
from ..math import __all__ as _new_all

__all__ = list(_new_all)


def __getattr__(name: str):
    """新模块以后新增的名字，这里也能自动取到。"""
    return getattr(_current, name)


warnings.warn(
    "llm_kernel.math_ 已更名为 llm_kernel.math，请改用新名字（见 docs/guide/migration.md）",
    DeprecationWarning,
    stacklevel=2,
)
