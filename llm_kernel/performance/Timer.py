"""兼容层：``llm_kernel.performance.Timer`` 这个**模块**已更名为 ``llm_kernel.performance.timing``。

改名理由：原来模块名（``Timer``）和它里面的类名（``Timer``）一样，谁先被导入谁盖住谁，
``from llm_kernel.performance import Timer`` 有时拿到的是类、有时拿到的是模块。

常用的 ``from llm_kernel.performance import Timer`` 不用改，受影响的是少数直接按模块路径导入的写法。
"""

import warnings

from .timing import Timer, timer  # noqa: F401

__all__ = ["Timer", "timer"]

warnings.warn(
    "llm_kernel.performance.Timer 模块已更名为 llm_kernel.performance.timing，请改用新路径",
    DeprecationWarning,
    stacklevel=2,
)
