# `llm_kernel.performance` — 计时

只依赖标准库。

```python
from llm_kernel.performance import Timer, timer

with Timer() as t:
    sum(range(10**6))

t.last, t.sum(), t.avg()
```

模块实现放在 `performance.timing` 里。**为什么改名字**：0.1.x 里模块名（`Timer`）
和它里面的类名（`Timer`）一样，谁先被导入谁盖住谁，
`from llm_kernel.performance import Timer` 有时拿到类、有时拿到模块。
旧路径 `llm_kernel.performance.Timer` 仍然可用（会提示弃用），1.0 移除。

::: llm_kernel.performance.timing
    options:
      members_order: source
