"""训练循环与累加器（需要 torch）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable

from ..deps import require
from .metrics import METRICS

if TYPE_CHECKING:  # 只是给类型检查器和文档看，运行时不导入 torch
    import torch

__all__ = ["Accumulator", "train_epoch"]


class Accumulator:
    """在 n 个变量上逐次累加（d2l 里那个类，补齐了 ``len`` / 迭代 / ``values``）。

    Examples:
        >>> from llm_kernel.nn import Accumulator
        >>> acc = Accumulator(2)
        >>> acc.add(1.0, 3.0)
        >>> acc.add(2.0, 5.0)
        >>> acc[0], acc[1]
        (3.0, 8.0)
        >>> len(acc), acc.values
        (2, [3.0, 8.0])
        >>> acc.reset()
        >>> acc.values
        [0.0, 0.0]
    """

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError(f"n 必须为正整数，收到 n={n!r}")
        self.data: list = [0.0] * n

    def add(self, *args) -> None:
        """把一组值和已有的值逐项相加；个数必须和 ``n`` 一致。"""
        if len(args) != len(self.data):
            raise ValueError(f"这个累加器有 {len(self.data)} 项，收到 {len(args)} 个值")
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self) -> None:
        """全部清零，便于开始下一个 epoch。"""
        self.data = [0.0] * len(self.data)

    @property
    def values(self) -> list:
        """当前的累加值（副本，改它不影响累加器）。"""
        return list(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self) -> str:
        return f"Accumulator({self.data})"


def train_epoch(
    net: Callable[[Any], torch.Tensor],
    train_iter: Iterable[tuple[Any, Any]],
    loss: Callable[..., Any] | str,
    updater: Any,
    *,
    metrics: Iterable[str | Callable[[Any, Any], float]] = ("accuracy",),
) -> dict[str, float]:
    """训练一个 epoch，返回本轮各项指标的字典。

    Args:
        net: 可调用的模型；如果是 ``torch.nn.Module``，会自动切到 ``net.train()``。
        train_iter: 迭代出 ``(X, y)`` 的批数据迭代器。
        loss: 损失函数，签名 ``loss(y_hat, y)``。可以传 :data:`llm_kernel.nn.LOSSES` 里注册的名字，
            也可以传任何 callable——包括 ``torch.nn.CrossEntropyLoss()`` 这类已经归约好的实例。
        updater: 优化器。``torch.optim.Optimizer`` 实例走 ``zero_grad()/step()``；
            其他可调用对象按 ``updater(batch_size)`` 调用（例如 :func:`llm_kernel.nn.sgd` 的闭包）。
        metrics: 指标集合，元素可以是 :data:`llm_kernel.nn.METRICS` 里注册的名字，也可以是自定义
            callable；默认 ``("accuracy",)``，回归任务传 ``[]`` 就别算指标。

    Returns:
        dict: ``{"loss": 整轮的平均损失, <指标名>: 整轮平均值}``。返回字典而不是裸 tuple，
        是为了加指标时不必改调用方的解包代码。

    Raises:
        ValueError: ``train_iter`` 没产出任何样本。
        KeyError: ``metrics`` 里写了没注册的名字。
        llm_kernel.deps.MissingDependencyError: 环境里没有 torch。

    Examples:
        >>> import torch
        >>> from llm_kernel.nn import sgd, squared_loss, train_epoch
        >>> X, y = torch.tensor([[1.0]]), torch.tensor([2.0])
        >>> w = torch.zeros(1, requires_grad=True)
        >>> def updater(batch_size):
        ...     sgd([w], lr=0.1, batch_size=batch_size)
        >>> round(train_epoch(lambda X: X @ w, [(X, y)], squared_loss, updater, metrics=[])["loss"], 4)
        2.0
    """
    torch = require("torch")
    from .losses import LOSSES  # 放在函数里，避免模块级循环导入

    loss_fn = LOSSES.resolve(loss)[1]
    metric_fns = [METRICS.resolve(m) for m in metrics]

    if isinstance(net, torch.nn.Module):
        net.train()

    total_loss = 0.0
    totals = {name: 0.0 for name, _ in metric_fns}
    num_examples = 0

    for X, y in train_iter:
        y_hat = net(X)
        batch_loss = loss_fn(y_hat, y)
        num = int(y.numel())
        reduced = batch_loss.dim() == 0

        # 统计损失时先 detach：只要数值，不要 autograd 图（否则会有警告）
        detached = batch_loss.detach()
        total_loss += float(detached) * num if reduced else float(detached.sum())

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            # 用内置优化器时，梯度按 batch 平均（和它的语义一致）
            (batch_loss if reduced else batch_loss.mean()).backward()
            updater.step()
        else:
            # 用自定义优化器（如 sgd）时，梯度保持 batch 的和，由优化器自己除以 batch_size
            (batch_loss if reduced else batch_loss.sum()).backward()
            updater(X.shape[0])

        for name, fn in metric_fns:
            totals[name] += float(fn(y_hat, y)) * num
        num_examples += num

    if num_examples == 0:
        raise ValueError("train_iter 一个样本都没产出，检查一下数据迭代器")

    result = {"loss": total_loss / num_examples}
    result.update({name: value / num_examples for name, value in totals.items()})
    return result
