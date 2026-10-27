"""二维曲线图（需要 matplotlib）。

只依赖 matplotlib，且是**可选依赖**：``import llm_kernel.plot`` 不需要装它，
调用函数时才检查。装法：``pip install "llm_kernel[plot]"``。
"""

from __future__ import annotations

from typing import Any, Sequence

from ..deps import require

__all__ = ["curve"]

#: 默认图尺寸，和 0.1.x 的 ``vector`` 保持一致。
DEFAULT_FIGSIZE = (10, 6)


def _check_same_length(x, y) -> None:
    try:
        len_x, len_y = len(x), len(y)
    except TypeError as exc:
        raise TypeError("x 和 y 必须是可求长度的序列，例如 list / numpy.ndarray / torch.Tensor") from exc
    if len_x != len_y:
        raise ValueError(f"x 和 y 的长度必须相同，收到 x 有 {len_x} 个、y 有 {len_y} 个")


def curve(
    x: Sequence[float],
    y: Sequence[float],
    *,
    label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    grid: bool = True,
    show: bool = True,
    ax: Any = None,
    **plot_kwargs: Any,
) -> tuple[Any, Any]:
    """画一条二维折线，返回 ``(fig, ax)``。

    与 0.1.x 的 ``vector`` 相比：返回图形对象、只在给了 ``label`` 时才画图例、
    可以传 ``ax`` 叠加多条曲线、可以 ``show=False`` 在 notebook 里接着改。

    Args:
        x (Sequence): 横坐标序列。
        y (Sequence): 纵坐标序列，长度必须和 ``x`` 相同。
        label (str | None): 曲线名字；只有给了才画图例。
        xlabel (str | None): 横轴标题。
        ylabel (str | None): 纵轴标题。
        title (str | None): 图标题。
        figsize (tuple): 新图的尺寸（传了 ``ax`` 时忽略）。
        grid (bool): 是否画网格。
        show (bool): 是否立刻 ``plt.show()``；想接着在同一坐标系上画，传 ``False``。
        ax (matplotlib.axes.Axes | None): 已有的坐标系，传了就把曲线画在它上面。
        **plot_kwargs: 透传给 ``ax.plot``，例如 ``color="b"``、``linewidth=2``、``linestyle="--"``。

    Returns:
        tuple: ``(fig, ax)``，即 ``matplotlib.figure.Figure`` 与 ``matplotlib.axes.Axes``。

    Raises:
        ValueError: ``x`` / ``y`` 长度不一致。
        TypeError: ``x`` / ``y`` 不是序列。
        llm_kernel.deps.MissingDependencyError: 环境里没有 matplotlib。

    Examples:
        >>> from llm_kernel.plot import curve
        >>> fig, ax = curve([1, 2, 3], [1, 4, 9], label="平方", show=False)
        >>> ax.get_xlabel(), len(ax.get_lines())
        ('', 1)
        >>> ax.get_legend().get_texts()[0].get_text()
        '平方'
    """
    require("matplotlib", extra="plot")
    from matplotlib import pyplot as plt

    _check_same_length(x, y)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(x, y, label=label, **plot_kwargs)
    if label is not None:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)
    if show:
        plt.show()
    return fig, ax


def vector(x: Sequence[float], y: Sequence[float], **kwargs: Any) -> tuple[Any, Any]:
    """0.1.x 的旧名字，等价于 ``curve(x, y, color="b", linewidth=2)``。

    保留它 0.1.x 的观感（蓝色粗线），同时把参数转发给 :func:`curve`，
    所以 ``vector(x, y, show=False)`` 这种新写法也能用。

    .. deprecated::
        请改用 :func:`curve`；``vector`` 会在 1.0 移除。
    """
    kwargs.setdefault("color", "b")
    kwargs.setdefault("linewidth", 2)
    return curve(x, y, **kwargs)
