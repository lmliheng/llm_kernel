"""正态分布（以及后续的概率分布）。

只依赖 numpy；符合库的统一约定：**标量进标量出、数组进 numpy 数组出**。
"""

from __future__ import annotations

import math as _math
import numbers
from typing import Sequence

import numpy as np

__all__ = ["normal"]


def _as_real(value, name: str) -> float:
    """把参数校验成实数并返回 float；bool 不算实数（True 当 1 用是 bug 温床）。"""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} 必须是实数，收到 {type(value).__name__}: {value!r}")
    return float(value)


def normal(
    x: float | Sequence[float] | np.ndarray,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> float | np.ndarray:
    """正态分布的概率密度。

    公式：``f(x) = 1 / (sigma * sqrt(2*pi)) * exp(-((x - mu) / sigma) ** 2 / 2)``。

    Args:
        x (float | list | tuple | numpy.ndarray | torch.Tensor): 自变量，标量或数组都可以。
            torch 张量会先按 numpy 数组计算（所以不在 ``math`` 里依赖 torch）。
        mu (float): 均值。默认 0.0（标准正态）。
        sigma (float): 标准差，必须为正数。默认 1.0（标准正态）。

    Returns:
        float | numpy.ndarray: 标量进就返回 Python ``float``（可直接 json 序列化）；
        数组进就返回与输入同形状的 ``numpy.ndarray``。

    Raises:
        TypeError: ``mu`` / ``sigma`` 不是实数，或 ``x`` 不是标量也不是能转成数组的对象。
        ValueError: ``sigma <= 0``。

    Examples:
        >>> from llm_kernel.math import normal
        >>> round(normal(0.0), 6)
        0.398942
        >>> normal([-1.0, 0.0, 1.0]).round(6).tolist()
        [0.241971, 0.398942, 0.241971]
        >>> normal(1.0, mu=1.0, sigma=2.0).__round__(6)
        0.199471
    """
    mu = _as_real(mu, "mu")
    sigma = _as_real(sigma, "sigma")
    if sigma <= 0:
        raise ValueError(f"sigma 必须为正数，收到 sigma={sigma!r}")

    factor = 1.0 / (sigma * _math.sqrt(2.0 * _math.pi))

    if isinstance(x, numbers.Real) and not isinstance(x, bool):
        z = (float(x) - mu) / sigma
        return factor * _math.exp(-0.5 * z * z)

    try:
        xs = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "x 必须是标量，或能被转成数值数组（list / tuple / numpy.ndarray / torch.Tensor），"
            f"收到 {type(x).__name__}"
        ) from exc

    z = (xs - mu) / sigma
    return factor * np.exp(-0.5 * z * z)
