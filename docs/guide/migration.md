# 从 0.1.x 迁移

0.2.0 重整了目录与命名，**旧名字全部保留**：`math_`、`plot_`、`import_`、`NeuralNetwork`
以及 `performance.Timer` 模块都是薄兼容层，导入时提示一次 `DeprecationWarning`，
计划在 1.0 删除。所以老 notebook 不会立刻坏，可以慢慢改。

## 名字对照

| 0.1.x | 0.2.0 推荐写法 | 说明 |
| --- | --- | --- |
| `from llm_kernel.math_ import normal` | `from llm_kernel.math import normal` | `math_` 的下划线是为了躲标准库，现在包内不需要 |
| `from llm_kernel.plot_ import vector` | `from llm_kernel.plot import curve` | `vector` 名字不达意（画的是折线）；旧名保留且观感不变 |
| `from llm_kernel.import_ import isImport` | `from llm_kernel.deps import require, require_all, is_installed, version_of` | 新函数不再往 stdout 打印，也不会抛裸 `Exception` |
| `from llm_kernel.NeuralNetwork import softmax, relu, sgd, accuracy, cross_entropy_loss` | `from llm_kernel.nn import ...` | 同名前缀统一成 `nn` |
| `from llm_kernel.performance import Timer, timer` | 不变 | 实现挪到 `performance.timing`，导入方式不变 |
| `from llm_kernel.performance.Timer import Timer` | `from llm_kernel.performance import Timer` | 原来模块名和类名都叫 `Timer`，会互相盖住 |
| `from llm_kernel import d2l` | 不变 | 现在 pip 装完也能用（0.1.0 的发行物里没有它） |
| `train_1(...)`（旧的训练函数） | `train_epoch(...)` | 返回字典，见下 |

## 行为变化

### 1. `normal` 更宽容，也更守规矩

```python
normal([-1.0, 0.0, 1.0])   # 0.1.x: TypeError（list 不能和 int 相减）
                           # 0.2.0: array([0.24197072, 0.39894228, 0.24197072])

type(normal(0.0))          # 0.1.x: numpy.float64
                           # 0.2.0: float（可直接 json 序列化）
```

`mu` / `sigma` 现在必须是真的实数：`sigma="1"` 会 `TypeError`；
`sigma=0` 由抛裸 `Exception` 改成 `ValueError`，消息里带上实际值。

### 2. `Timer` 更结实

| 场景 | 0.1.x | 0.2.0 |
| --- | --- | --- |
| 空的 `avg()` | `ZeroDivisionError` | `ValueError: 还没有任何计时结果…` |
| 未 `start()` 就 `stop()` | `AttributeError: 'Timer' object has no attribute 'tik'` | `RuntimeError: 还没有 start()，不能 stop()…` |
| 计时时钟 | `time.time()`（不单调） | `time.perf_counter()`（单调、分辨率更高） |
| `cumsum()` | numpy 数组转 list | `itertools.accumulate`，纯标准库 |
| 新能力 | — | `with Timer() as t`、`t.last`、`t.count`、`t.elapsed`、`t.reset()`、`@timer(verbose=False)` |

`@timer` 的输出格式没变；另外现在**异常也会被计时**（用 `finally`）。

### 3. `plot.vector` → `plot.curve`

```python
# 0.1.x：画完就 plt.show()，什么也拿不到
vector(x, y)

# 0.2.0：拿到 fig / ax，可以叠加、保存、继续改
fig, ax = curve(x, y, label="y", show=False)
curve(x2, y2, label="y2", ax=ax)
```

旧名字 `vector(x, y)` 仍然可用，观感也保持原样（蓝色粗线），并且支持新的关键字参数。
0.1.x 里"没给 label 却调 `plt.legend()`"产生的警告已经修掉。

### 4. `cross_entropy_loss` → `cross_entropy`

```python
cross_entropy(y_hat, y)                        # 逐样本损失向量（和 0.1.x 一样）
cross_entropy(y_hat, y, reduction="mean")      # 新增：先归约
```

输入仍然是**概率**（先 `softmax`），这一点在 0.1.x 的文档里完全没写，现在写清楚了。
名字改成 `cross_entropy` 是为了和 `torch.nn.functional.cross_entropy` 对齐；
旧名字 `cross_entropy_loss` 仍指向同一个函数。

### 5. 训练循环返回字典

```python
# 0.1.x
train_loss, train_acc = train_1(net, train_iter, loss, updater)

# 0.2.0
result = train_epoch(net, train_iter, loss, updater)          # {'loss': …, 'accuracy': …}
result["loss"], result["accuracy"]
train_epoch(net, data, loss, updater, metrics=["accuracy", "rmse"])   # 想加多少加多少
train_epoch(net, data, loss, updater, metrics=[])                     # 只要损失
```

顺带修好的问题：`NeuralNetwork/train.py` 原先有一句 `from accuracy import accuracy`
（顶层绝对导入），一 import 就 `ModuleNotFoundError`，所以那个 `Accumulator` 和训练函数
在 0.1.x 里其实用不了。现在 `Accumulator` 在 `llm_kernel.nn` 里，并且补了
`len()` / 迭代 / `values` / `reset()`。

### 6. `isImport` → `deps` 系列

```python
# 0.1.x：会 print，缺依赖时抛 Exception
isImport(["torch", "numpy"])

# 0.2.0
from llm_kernel.deps import require_all
require_all(["torch", "numpy"])     # {'torch': '2.14.1+cpu', 'numpy': '2.5.3'}（不打印）
```

`import_` 里的 `isImport()` 仍然保留原行为，方便老 notebook。

## 迁移检查清单

- [ ] `from llm_kernel.math_ import …` → `from llm_kernel.math import …`
- [ ] `from llm_kernel.plot_ import vector` → `from llm_kernel.plot import curve`
- [ ] `from llm_kernel.NeuralNetwork import …` → `from llm_kernel.nn import …`
- [ ] `from llm_kernel.import_ import isImport` → `from llm_kernel.deps import require_all`
- [ ] 自己写的训练循环 → 用 `train_epoch()`，返回值按字典解包
- [ ] 把 `import llm_kernel.xxx` 的旧名字跑一遍，确认没有新报错（会有 `DeprecationWarning`）
- [ ] 老代码里如果依赖"`avg()` 返回 nan / 报 ZeroDivisionError"，改掉

!!! tip "怎么看到弃用提醒"
    `DeprecationWarning` 默认被 Python 隐藏。想看就加一句：

    ```python
    import warnings
    warnings.simplefilter("always", DeprecationWarning)
    ```
