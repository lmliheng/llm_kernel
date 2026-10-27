"""兼容层：``llm_kernel.plot_`` 已更名为 :mod:`llm_kernel.plot`。

这里只做转发 + 提醒一次，1.0 版本删除。
"""

import warnings

from .. import plot as _current
from ..plot import *  # noqa: F401,F403
from ..plot import __all__ as _new_all

__all__ = list(_new_all)


def __getattr__(name: str):
    """新模块以后新增的名字，这里也能自动取到。"""
    return getattr(_current, name)


warnings.warn(
    "llm_kernel.plot_ 已更名为 llm_kernel.plot，请改用新名字（见 docs/guide/migration.md）",
    DeprecationWarning,
    stacklevel=2,
)
