# 安装与上手

## 安装

```bash
pip install llm_kernel
```

核心包只依赖 **numpy**。需要 torch 或 matplotlib 的功能用 extra 单独装：

| 命令 | 装了什么 | 对应模块 |
| --- | --- | --- |
| `pip install llm_kernel` | numpy | `math`、`performance`、`deps` |
| `pip install "llm_kernel[nn]"` | torch | `nn` |
| `pip install "llm_kernel[plot]"` | matplotlib | `plot` |
| `pip install "llm_kernel[d2l]"` | torch、torchvision、pandas、requests、ipython、matplotlib | `d2l` |
| `pip install "llm_kernel[dev]"` | pytest、build、twine | 开发用 |
| `pip install "llm_kernel[docs]"` | mkdocs-material、mkdocstrings | 写文档用 |

## 为什么要有 extra

只想用 `math.normal()` 的人，不该被迫装 200MB 的 torch。所以：

```python
import llm_kernel as lk      # 不导入 numpy / torch / matplotlib
lk.math.normal(0.0)          # 这时才导入 numpy
```

`import llm_kernel.nn` 在没有 torch 的机器上也能成功——真正的检查推迟到调用函数时：

```python
>>> import llm_kernel
>>> llm_kernel.nn.relu([1, 2])
Traceback (most recent call last):
    ...
llm_kernel.deps.MissingDependencyError: 缺少可选依赖 'torch'，无法使用这个功能；安装方式：pip install "llm_kernel[nn]"
```

这个异常继承自 `ImportError`，所以老的 `except ImportError` 也能接住；想单独处理就写
`except llm_kernel.deps.MissingDependencyError`。

也可以主动检查，避免在用户面前抛异常：

```python
from llm_kernel.deps import is_installed, require, require_all, version_of

is_installed("torch")        # True / False
version_of("numpy")          # '2.5.3'；不是发行包或没装时返回 None
require_all(["numpy"])       # {'numpy': '2.5.3'}
```

## 上手

### 数学

```python
from llm_kernel.math import normal

normal(0.0)                        # 0.3989422804014327（Python float，可直接 json 序列化）
normal([-1.0, 0.0, 1.0])           # numpy 数组
normal(1.0, mu=1.0, sigma=2.0)     # 换个均值和标准差
```

`sigma` 必须是正数，否则 `ValueError`；`mu` / `sigma` 类型不对会 `TypeError`，
报错里带上你实际传进去的值。

### 计时

```python
from llm_kernel.performance import Timer, timer

t = Timer()
t.start()
# ... 要测量的代码 ...
t.stop()

t.last        # 最近一次
t.sum()       # 合计
t.avg()       # 平均
t.cumsum()    # 累计序列

with Timer() as t:            # 也可以这样，自动记一笔
    ...

@timer                        # 装饰器：打印耗时
def work():
    ...

@timer(verbose=False)         # 静默版：耗时挂在 work.times / work.last 上
def quiet_work():
    ...
```

### 神经网络

```python
import torch
from llm_kernel.nn import sgd, softmax, squared_loss, train_epoch

X = torch.randn(64, 2)
y = X @ torch.tensor([[3.0], [-2.0]]) + 1.5
data = [(X[i:i+16], y[i:i+16]) for i in range(0, 64, 16)]

w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
net = lambda X: X @ w + b

def updater(batch_size):
    sgd([w, b], lr=0.05, batch_size=batch_size)

for epoch in range(30):
    print(train_epoch(net, data, squared_loss, updater, metrics=[]))
# {'loss': 7.87...} → ... → {'loss': 0.0008...}
```

`train_epoch()` 返回字典而不是一堆裸数字，所以**加指标不用改解包代码**：

```python
from llm_kernel.nn import METRICS

@METRICS.register("rmse")
def rmse(y_hat, y):
    return float((y_hat - y).pow(2).mean().sqrt())

train_epoch(net, data, squared_loss, updater, metrics=["rmse"])
# {'loss': ..., 'rmse': ...}
```

### 画图

```python
from llm_kernel.plot import curve

fig, ax = curve([1, 2, 3], [1, 4, 9], label="平方", xlabel="x", ylabel="y")
curve([1, 2, 3], [1, 2, 3], label="线性", ax=ax)   # 叠在同一条坐标系上
ax.set_ylim(0, 10)                                  # 拿得到 ax，想怎么改都行
```

## 常见问题

??? question "Notebook 里图不显示 / 卡住"
    `curve()` 默认 `show=True`（立刻 `plt.show()`）。想接着改图，传 `show=False`，
    最后在单元格末尾让它自然显示，或者手动调 `plt.show()`。

??? question "服务器上没有显示器"
    用无界面后端即可：在导入 matplotlib 之前设置环境变量 `MPLBACKEND=Agg`
    （本项目的测试就是这么做的，见 `tests/conftest.py`）。

??? question "装 torch 太慢 / 想要 CPU 版"
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```
    再 `pip install "llm_kernel[nn]"`（torch 已经在了，这一步不会重复下载）。

??? question "支持哪些 Python 版本"
    3.9 及以上。CI 会在 3.9 / 3.12 / 3.13 上跑测试。
