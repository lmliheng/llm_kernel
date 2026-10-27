# `llm_kernel.math` — 数学与概率计算

只依赖 numpy。输入输出约定：**标量进标量出（Python `float`）、数组进 `numpy.ndarray` 出**，
`list` / `tuple` / `numpy.ndarray` / `torch.Tensor` 都能直接传。

```python
from llm_kernel.math import normal

normal(0.0)                      # 0.3989422804014327
normal([-1.0, 0.0, 1.0])         # array([0.24197072, 0.39894228, 0.24197072])
normal(1.0, mu=1.0, sigma=2.0)   # 0.19947114020071638
```

新增函数的方式见 [API 设计说明](../guide/design.md#extension-recipes)。

## 概率分布

::: llm_kernel.math.distributions
    options:
      members:
        - normal
