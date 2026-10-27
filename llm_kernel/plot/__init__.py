"""可视化（需要 matplotlib，装法：``pip install "llm_kernel[plot]"``）。

``curve`` 是正式名字；``vector`` 是 0.1.x 的旧名字，保留到 1.0。
"""

from . import curves
from .curves import curve, vector

__all__ = ["curve", "curves", "vector"]
