# `llm_kernel.plot` — 可视化

需要 matplotlib（`pip install "llm_kernel[plot]"`）。

```python
from llm_kernel.plot import curve

fig, ax = curve([1, 2, 3], [1, 4, 9], label="平方", xlabel="x", ylabel="y")
curve([1, 2, 3], [1, 2, 3], label="线性", ax=ax)   # 叠在同一条坐标系上
ax.set_ylim(0, 10)
```

`curve()` 返回 `(fig, ax)`，所以画完之后还能继续加工；`show=False` 不会立刻弹窗，
方便在 notebook 里叠加多条曲线。

0.1.x 的旧名字 `vector(x, y)` 仍然可用，观感和以前一样（蓝色粗线），
并支持新的关键字参数。它会在 1.0 移除。

::: llm_kernel.plot.curves
    options:
      members_order: source
