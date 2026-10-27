"""优化器（需要 torch）。

约定：签名 ``optimizer(params, lr, batch_size)``，**参数是参数张量的可迭代对象**，
这样既能配合 :func:`llm_kernel.nn.train_epoch` 用（它把 batch 大小传进来），
也能自己手写循环时直接调用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from ..deps import require

if TYPE_CHECKING:  # 只是给类型检查器和文档看，运行时不导入 torch
    import torch

__all__ = ["sgd"]


def sgd(params: Iterable[torch.Tensor], lr: float, batch_size: int) -> None:
    """小批量随机梯度下降。

    按 d2l 的约定，调用前梯度里累加的是**整个 batch 的和**，所以这里除以 ``batch_size``
    得到平均梯度；更新后把每个参数的梯度清零。

    Args:
        params (Iterable[torch.Tensor]): 待更新的参数张量（通常一个模型一层就一个）。
        lr (float): 学习率，必须为正数。
        batch_size (int): 本批样本数，必须为正整数。

    Returns:
        None: 原地更新参数。

    Raises:
        ValueError: ``lr <= 0`` 或 ``batch_size <= 0``。
        llm_kernel.deps.MissingDependencyError: 环境里没有 torch。

    Examples:
        >>> import torch
        >>> from llm_kernel.nn import sgd
        >>> w = torch.tensor([1.0], requires_grad=True)
        >>> (w * 2).sum().backward()      # 梯度是 batch 的和
        >>> sgd([w], lr=0.1, batch_size=2)
        >>> w.item().__round__(4)
        0.9
    """
    torch = require("torch")
    if lr <= 0:
        raise ValueError(f"lr 必须为正数，收到 lr={lr!r}")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，收到 batch_size={batch_size!r}")
    with torch.no_grad():
        for param in params:
            if param.grad is None:
                raise ValueError("参数还没有梯度，请先对损失调用 backward()")
            param -= lr * param.grad / batch_size
            param.grad.zero_()
