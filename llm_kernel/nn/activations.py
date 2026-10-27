"""激活函数（需要 torch）。

约定：``f(x) -> torch.Tensor``，不修改输入，返回新张量。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..deps import require

if TYPE_CHECKING:  # 只是给类型检查器和文档看，运行时不导入 torch
    import torch

__all__ = ["relu", "softmax"]


def relu(x: torch.Tensor) -> torch.Tensor:
    """逐元素 ReLU：``max(x, 0)``。

    等价于 ``torch.nn.functional.relu``，但用起来不用先 import torch。

    Args:
        x (torch.Tensor): 任意形状和数值类型的张量。

    Returns:
        torch.Tensor: 与 ``x`` 同形状的新张量（整数张量保持整数）。

    Raises:
        llm_kernel.deps.MissingDependencyError: 环境里没有 torch。

    Examples:
        >>> import torch
        >>> from llm_kernel.nn import relu
        >>> relu(torch.tensor([-1.0, 0.0, 2.0]))
        tensor([0., 0., 2.])
    """
    torch = require("torch")
    return torch.clamp(x, min=0)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """沿指定维度做 softmax（先减去最大值，避免 ``exp`` 溢出）。

    和 ``torch.nn.functional.softmax(x, dim=dim)`` 结果一致。

    Args:
        x (torch.Tensor): 分数张量，形状里最后一维通常就是类别维。
        dim (int): 归一化的维度，默认 ``-1``。

    Returns:
        torch.Tensor: 与 ``x`` 同形状、沿 ``dim`` 求和为 1 的张量。

    Raises:
        llm_kernel.deps.MissingDependencyError: 环境里没有 torch。

    Examples:
        >>> import torch
        >>> from llm_kernel.nn import softmax
        >>> softmax(torch.tensor([[1.0, 2.0, 3.0]])).sum().round(decimals=4)
        tensor(1.)
        >>> softmax(torch.tensor([1000.0, 1000.0])).tolist()
        [0.5, 0.5]
    """
    torch = require("torch")
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)
