"""损失函数（需要 torch）。

统一约定：``loss(y_hat, y) -> torch.Tensor``，**默认返回逐样本的损失向量**（d2l 的习惯），
由调用方自己决定怎么归约；想省事就传 ``reduction="mean"`` / ``"sum"``。

要求"返回逐样本向量"是有意的：训练时既要用 ``sum()`` 反传（配合自定义优化器），
又要用均值做报告，归约方式不该写死在损失函数里。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..deps import require
from .registry import Registry

if TYPE_CHECKING:  # 只是给类型检查器和文档看，运行时不导入 torch
    import torch

__all__ = ["LOSSES", "cross_entropy", "cross_entropy_loss", "squared_loss"]

LOSSES = Registry("损失函数")
"""已注册的损失函数；``@LOSSES.register("my_loss")`` 就能把自己的损失接进来。"""


@LOSSES.register("cross_entropy")
def cross_entropy(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    *,
    reduction: str = "none",
) -> torch.Tensor:
    """交叉熵损失（输入是**概率**，不是 logits）。

    Args:
        y_hat (torch.Tensor): 形状 ``(n, k)`` 的概率，例如 :func:`llm_kernel.nn.softmax` 的输出。
        y (torch.Tensor): 形状 ``(n,)`` 的真实类别下标（``0 <= y < k``）。
        reduction (str): ``"none"``（默认，逐样本）/ ``"mean"`` / ``"sum"``。

    Returns:
        torch.Tensor: ``reduction="none"`` 时形状为 ``(n,)``；否则是标量张量。

    Raises:
        ValueError: ``reduction`` 不是那三个值之一。
        llm_kernel.deps.MissingDependencyError: 环境里没有 torch。

    Examples:
        >>> import torch
        >>> from llm_kernel.nn import cross_entropy
        >>> p = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        >>> cross_entropy(p, torch.tensor([0, 1])).round(decimals=4)
        tensor([0.1054, 0.2231])
        >>> cross_entropy(p, torch.tensor([0, 1]), reduction="mean").round(decimals=4)
        tensor(0.1643)
    """
    torch = require("torch")
    losses = -torch.log(y_hat[range(len(y_hat)), y])
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError(f'reduction 只能是 "none" / "mean" / "sum"，收到 {reduction!r}')


cross_entropy_loss = cross_entropy
"""0.1.x 的旧名字，等价于 :func:`cross_entropy`。"""


@LOSSES.register("squared_loss")
def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """平方损失 ``(y_hat - y) ** 2 / 2``，逐样本返回。

    注意是**除以 2** 的 d2l 版本（求导后系数正好是 1），不要和 MSE 混用。

    Args:
        y_hat (torch.Tensor): 预测值。
        y (torch.Tensor): 真实值，会被 reshape 成 ``y_hat`` 的形状。

    Returns:
        torch.Tensor: 与 ``y_hat`` 同形状的损失。

    Raises:
        llm_kernel.deps.MissingDependencyError: 环境里没有 torch。

    Examples:
        >>> import torch
        >>> from llm_kernel.nn import squared_loss
        >>> squared_loss(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0]))
        tensor([0., 2.])
    """
    torch = require("torch")
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
