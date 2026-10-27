# llm_kernel

张量计算、机器学习、深度学习与模型训练推理的可复用函数与最佳实践——边学边写、持续补充。

目标是**小而清楚**：核心只依赖 numpy，torch / matplotlib 按需安装；每个函数都能单独看懂，
每段文档里的示例都被测试真的跑过。

## 安装

```bash
pip install llm_kernel              # 核心：只依赖 numpy
pip install "llm_kernel[nn]"        # 需要 torch 的功能（nn 模块）
pip install "llm_kernel[plot]"      # 画图（matplotlib）
pip install "llm_kernel[d2l]"       # notebook 里用内置的 d2l 工具库
```

## 30 秒上手

```python
import llm_kernel as lk

# 1. 数学：标量进标量出，数组进 numpy 数组出
lk.math.normal(0.0)                  # 0.3989422804014327
lk.math.normal([-1.0, 0.0, 1.0])     # array([0.24197072, 0.39894228, 0.24197072])

# 2. 计时：手动或用 with
timer = lk.performance.Timer()
with timer:
    sum(range(10**6))
timer.last                            # 这一次的耗时（秒）

# 3. 神经网络：只用 numpy 的环境里 import 也不会报错，调用时才检查 torch
import torch
logits = torch.tensor([[1.0, 2.0, 3.0]])
lk.nn.softmax(logits)                 # tensor([[0.0900, 0.2447, 0.6652]])

# 4. 画图：返回 (fig, ax)，可以接着改
fig, ax = lk.plot.curve([1, 2, 3], [1, 4, 9], label="平方", xlabel="x", ylabel="y")
```

## 模块

| 模块 | 内容 | 依赖 |
| --- | --- | --- |
| [`llm_kernel.math`](api/math.md) | 概率分布等数学函数，如 `normal` | numpy |
| [`llm_kernel.nn`](api/nn.md) | 激活、损失、指标、优化器、训练循环 | torch |
| [`llm_kernel.performance`](api/performance.md) | 计时器 `Timer`、装饰器 `timer` | 标准库 |
| [`llm_kernel.plot`](api/plot.md) | 二维折线 `curve` | matplotlib |
| [`llm_kernel.deps`](api/deps.md) | 可选依赖检查，缺依赖时给安装命令 | 标准库 |
| [`llm_kernel.d2l`](api/d2l.md) | 《动手学深度学习》配套工具库的副本 | torch、pandas 等 |

## 设计要点

- **一个入口**：`import llm_kernel as lk`，用 `lk.<领域>.` 取用；领域由模块表达，函数名里不出现领域词。
- **惰性加载**：`import llm_kernel` 不导入 numpy / torch / matplotlib，没装 torch 也能导入。
- **可选依赖只在调用点检查**：缺了会抛 `MissingDependencyError`，报错里带着
  `pip install "llm_kernel[nn]"` 这样的安装命令。
- **向后兼容**：0.1.x 的旧名字（`math_`、`plot_`、`import_`、`NeuralNetwork`）继续可用，
  访问时提醒一次，1.0 才移除。见[迁移指南](guide/migration.md)。

完整理由见 [API 设计说明](guide/design.md)。

## 兼容性

```python
# 0.1.x 的写法仍然能跑（会有一条 DeprecationWarning）
from llm_kernel.math_ import normal
from llm_kernel.NeuralNetwork import softmax, relu, sgd, accuracy, cross_entropy_loss
```

新代码请用 `llm_kernel.math` / `llm_kernel.nn`，对照表在[迁移指南](guide/migration.md)。
