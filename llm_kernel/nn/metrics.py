"""评价指标（需要 torch）。

统一约定：``metric(y_hat, y) -> float``，返回**本批**的指标（0~1 的比例），
由 :func:`llm_kernel.nn.train_epoch` 按样本数加权汇总成整个 epoch 的值。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..deps import require
from .registry import Registry

if TYPE_CHECKING:  # 只是给类型检查器和文档看，运行时不导入 torch
    import torch

__all__ = ["METRICS", "accuracy"]

METRICS = Registry("指标")
"""已注册的指标；``@METRICS.register("rmse")`` 就能把自己的指标接进来。"""


@METRICS.register("accuracy")
def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """分类精度：预测对的样本数 / 样本总数。

    ``y_hat`` 是 ``(n, k)`` 的得分或概率时先取 ``argmax``；已经是 ``(n,)`` 的类别下标时直接用。

    Args:
        y_hat (torch.Tensor): 形状 ``(n, k)`` 的预测或 ``(n,)`` 的预测类别。
        y (torch.Tensor): 形状 ``(n,)`` 的真实类别下标。

    Returns:
        float: 本批的正确率，取值 ``[0, 1]``。

    Raises:
        ValueError: ``y`` 是空的（避免除零这种模糊报错）。
        llm_kernel.deps.MissingDependencyError: 环境里没有 torch。

    Examples:
        >>> import torch
        >>> from llm_kernel.nn import accuracy
        >>> y_hat = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        >>> accuracy(y_hat, torch.tensor([1, 1, 1])).__round__(3)
        0.667
    """
    torch = require("torch")
    if y.numel() == 0:
        raise ValueError("y 不能为空，无法计算精度")
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    y = y.reshape(y_hat.shape)
    correct = y_hat.type(y.dtype) == y
    return float(correct.sum()) / y.numel()
