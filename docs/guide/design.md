# API 设计说明

这份文档回答一个问题：这个库为什么长这样？想知道"怎么用"请看[安装与上手](install.md)，
想改代码请看[开发与发布](develop.md)。

## 八条原则

### 1. 一个入口，领域分治

```python
import llm_kernel as lk
lk.math.normal(0.0)
lk.nn.softmax(logits)
lk.performance.Timer()
lk.plot.curve(x, y)
lk.deps.require("torch")
```

**规则**：函数名里不出现领域词，领域由模块表达。`math.normal` 而不是 `normal_math`，
`plot.curve` 而不是 `plot_vector_curve`。子包名用完整单词（`performance` 而不是 `perf`），
因为读者点开的是 `help()`，不是源码。

### 2. 惰性加载

`llm_kernel/__init__.py` 用 PEP 562 的模块级 `__getattr__`：`import llm_kernel` 不导入
numpy / torch / matplotlib，只有真的访问 `lk.nn` 时才 `import llm_kernel.nn`。

好处很实际：装不装 torch 都能 `import llm_kernel`；导入快；`import` 一个库不会连带
触发别的库的初始化副作用。

### 3. 可选依赖只在调用点检查

子模块顶部**不写** `import torch`，而是函数体第一行 `torch = require("torch")`。
缺依赖时光是 `import llm_kernel.nn` 不会失败，调用函数才抛
`MissingDependencyError(ImportError)`，而且报错消息里是能直接复制的安装命令。

为什么不用顶层 import 加 try/except：那样报错发生在 import 阶段，用户看到的是
`ModuleNotFoundError: No module named 'torch'`——他不会知道该装什么 extra，也不知道
自己是不是用错了库。

### 4. 命名有规矩

| 对象 | 规矩 | 例子 |
| --- | --- | --- |
| 函数 | `snake_case`，动词或名词 | `normal`、`cross_entropy`、`train_epoch` |
| 类 | `CapWords` | `Timer`、`Accumulator`、`Registry` |
| 模块 | `snake_case`，不留装饰性后缀 | `distributions`、`timing`、`curves`、`requirements` |
| 常量/注册表 | `UPPER_CASE` | `LOSSES`、`METRICS`、`DEFAULT_FIGSIZE` |
| 内部名字 | 一个下划线 | `_check_same_length`、`_INSTALL_HINT` |

同一概念跨模块用同一个词：损失就是 `loss`，不是 `cost`/`criterion`/`objective` 混着来。

### 5. 参数约定

- 被计算的位置放第一个参数：`normal(x, ...)`、`curve(x, y, ...)`。
- 配置项**关键字化 + 给默认值**：`normal(x, mu=0.0, sigma=1.0)` 直接就是标准正态。
- 给"要不要立刻出结果"的开关：`plot.curve(..., show=False)`、
  `cross_entropy(..., reduction="none")`。
- 参数名写全，不用缩写：`batch_size` 而不是 `bs`，`reduction` 而不是 `red`。

### 6. 输入输出规则统一

| 模块 | 输入 | 输出 |
| --- | --- | --- |
| `math` | 标量 / `list` / `tuple` / `numpy.ndarray` / `torch.Tensor` | 标量进 → Python `float`；数组进 → `numpy.ndarray` |
| `nn` | torch 张量，形状约定写进 docstring | torch 张量 |
| `plot` | 任意序列 | `(fig, ax)` |
| `deps` | 模块名 / 发行包名 | `bool` / `str \| None` / 模块对象 |

"标量进标量出、数组进数组出"这条尤其重要：它让 `normal(0.0)` 能直接进 json，
也让 `normal(xs)` 能直接喂给 matplotlib。

### 7. 错误只说三件事：哪里错、收到什么、怎么改

- **参数值不对** → 内置 `ValueError`：`sigma 必须为正数，收到 sigma=-1.0`
- **参数类型不对** → 内置 `TypeError`：`sigma 必须是实数，收到 str: '1'`
- **调用顺序不对** → 内置 `RuntimeError`：`还没有 start()，不能 stop()；请先调用 t.start()`
- **缺少可选依赖** → `deps.MissingDependencyError`（继承 `ImportError`）

不自定义一堆异常类，也不抛裸 `Exception`——调用方要能一眼看出该 `except` 什么。

### 8. 扩展性靠注册表 + 鸭子类型

不搞插件系统，一个约 30 行的 `Registry` 就够：

```python
from llm_kernel.nn import METRICS, LOSSES

@METRICS.register("rmse")                 # 起个名字
def rmse(y_hat, y):
    return float((y_hat - y).pow(2).mean().sqrt())

@LOSSES.register                            # 用函数名当键
def my_loss(y_hat, y):
    ...

train_epoch(net, data, my_loss, updater, metrics=["accuracy", "rmse"])
```

关键是**两条路都通**：注册了就能按名字用，没注册也能直接传函数对象。
库不需要认识使用者的函数，使用者也不需要继承库里的任何东西。

## 扩展配方 { #extension-recipes }

### 加一个数学函数

1. 在 `llm_kernel/math/` 里找主题合适的模块（没有就新建，例如 `statistics.py`）。
2. 写函数：`snake_case`、对象放第一个参数、配置项关键字化、写 Google 风格 docstring
   加一条 `Examples:`。
3. 在 `llm_kernel/math/__init__.py` 里导出，并加进 `__all__`。
4. 在 `tests/` 里加用例；`tests/test_docstrings.py` 会自动跑你写的示例。

### 加一个损失或指标

只要符合签名 `loss(y_hat, y) -> Tensor`（逐样本向量）或 `metric(y_hat, y) -> float`，
就能直接传给 `train_epoch`。想按名字用就注册：

```python
from llm_kernel.nn import METRICS
METRICS.register("top3_accuracy", top3_accuracy)
```

注册名重复会 `ValueError`，取不存在的名字会 `KeyError` 并列出可选名字——写错名字时不用猜。

### 加一个新的子模块（例如 `llm_kernel.data`）

1. 建 `llm_kernel/data/__init__.py`（写模块级 docstring：这个领域是干什么的、依赖是什么）。
2. 在 `llm_kernel/__init__.py` 的 `_SUBMODULES` 里加上 `"data"`，并把 `__all__` 同步。
3. 需要可选依赖就用 `deps.require()`，并加进 `pyproject.toml` 的
   `[project.optional-dependencies]`。
4. 加文档页 `docs/api/data.md` 和 `mkdocs.yml` 的导航项。
5. 加测试。

### 调整已有行为

行为变化要写进 `docs/changelog.md`；破坏性改动必须在 minor 版本里说明，
并给旧名字留一个薄兼容层（见下一节）。

## 兼容与版本策略

- **0.x 阶段**：破坏性改动进 minor 版本（0.1 → 0.2），并在变更日志里写清楚。
- **旧名字保留**：`math_`、`plot_`、`import_`、`NeuralNetwork`、`performance.Timer`（模块）
  都是薄兼容层，导入时给一次 `DeprecationWarning`，**1.0 版本删除**。
- **版本号只有一个来源**：`llm_kernel/__init__.py` 里的 `__version__`，
  `pyproject.toml` 用 `dynamic = ["version"]` 读它，发布工作流会校验 tag 与它一致。
  这样不会再出现"pyproject 写 0.1.0、代码里写 0.1.1、PyPI 上是另一个"的情况。

## 刻意不做的几件事

- **不做全局配置对象**：没有 `llm_kernel.config`，每个函数的参数就是它的配置。
- **不做自动微分**：那是 torch 的事，本库只做"小而清楚"的胶水。
- **不封装 pandas**：需要表格就用 pandas，本库不提供半个 pandas。
- **不隐藏依赖**：装什么、缺什么、怎么装，都在报错里说清楚，不做"运行时偷偷 import"。
