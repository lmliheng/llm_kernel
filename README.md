# llm_kernel

张量计算、机器学习、深度学习与模型训练推理的可复用函数与最佳实践——边学边写、持续补充。

[![Tests](https://github.com/lmliheng/llm_kernel/actions/workflows/tests.yml/badge.svg)](https://github.com/lmliheng/llm_kernel/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-lmliheng.github.io-blue)](https://lmliheng.github.io/llm_kernel/)
[![PyPI](https://img.shields.io/pypi/v/llm_kernel)](https://pypi.org/project/llm_kernel/)
[![Python](https://img.shields.io/pypi/pyversions/llm_kernel)](https://pypi.org/project/llm_kernel/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Small, dependency-light helpers for tensor math, machine learning, deep learning and model
training / inference. 核心只依赖 numpy，torch / matplotlib 按需安装。

**文档：<https://lmliheng.github.io/llm_kernel/>**

## 安装 / Installation

```bash
pip install llm_kernel              # 核心：只依赖 numpy
pip install "llm_kernel[nn]"        # 需要 torch 的功能（llm_kernel.nn）
pip install "llm_kernel[plot]"      # 画图（matplotlib）
pip install "llm_kernel[d2l]"       # notebook 里用内置的 d2l 工具库
```

## 30 秒上手 / Quickstart

```python
import llm_kernel as lk

# 数学：标量进标量出，数组进 numpy 数组出
lk.math.normal(0.0)                    # 0.3989422804014327
lk.math.normal([-1.0, 0.0, 1.0])       # array([0.24197072, 0.39894228, 0.24197072])

# 计时：手动或用 with
timer = lk.performance.Timer()
with timer:
    sum(range(10**6))
timer.last                             # 这一次的耗时（秒）

# 神经网络：只用 numpy 的环境里 import 也不会报错，调用时才检查 torch
import torch
lk.nn.softmax(torch.tensor([[1.0, 2.0, 3.0]]))

# 画图：返回 (fig, ax)，可以接着改
fig, ax = lk.plot.curve([1, 2, 3], [1, 4, 9], label="平方", xlabel="x", ylabel="y")
```

## 模块 / Modules

| 模块 | 内容 | 依赖 |
| --- | --- | --- |
| `llm_kernel.math` | 概率分布等数学函数，如 `normal` | numpy |
| `llm_kernel.nn` | 激活、损失、指标、优化器、训练循环 | torch |
| `llm_kernel.performance` | 计时器 `Timer`、装饰器 `timer` | 标准库 |
| `llm_kernel.plot` | 二维折线 `curve` | matplotlib |
| `llm_kernel.deps` | 可选依赖检查，缺依赖时给出安装命令 | 标准库 |
| `llm_kernel.d2l` | 《动手学深度学习》配套工具库的副本 | torch、pandas 等 |

## 设计要点 / Design notes

- **一个入口**：`import llm_kernel as lk`，用 `lk.<领域>.` 取用；领域由模块表达。
- **惰性加载**：`import llm_kernel` 不导入 numpy / torch / matplotlib，没装 torch 也能导入。
- **可选依赖只在调用点检查**：缺依赖时抛 `MissingDependencyError`，消息里带着
  `pip install "llm_kernel[nn]"` 这样的安装命令。
- **扩展靠注册表 + 鸭子类型**：自定义损失/指标既能注册名字，也能直接传函数。

完整说明见[设计文档](https://lmliheng.github.io/llm_kernel/guide/design/)。

## 自定义损失与指标 / Custom losses and metrics

```python
from llm_kernel.nn import METRICS, squared_loss, train_epoch

@METRICS.register("rmse")
def rmse(y_hat, y):
    return float((y_hat - y).pow(2).mean().sqrt())

train_epoch(net, data, squared_loss, updater, metrics=["accuracy", "rmse"])
```

## 兼容性 / Compatibility

0.1.x 的旧名字（`math_`、`plot_`、`import_`、`NeuralNetwork`、`performance.Timer` 模块）
继续可用，导入时给一次 `DeprecationWarning`，1.0 版本移除：

```python
from llm_kernel.math_ import normal                     # 老写法还能跑
from llm_kernel.NeuralNetwork import softmax, relu      # 老写法还能跑
```

新代码请写 `from llm_kernel.math import normal`、`from llm_kernel.nn import softmax`。
对照表见[迁移指南](https://lmliheng.github.io/llm_kernel/guide/migration/)。

## 开发 / Development

```bash
git clone https://github.com/lmliheng/llm_kernel.git
cd llm_kernel
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"
pytest
mkdocs serve
```

发布前一定确认 wheel 里真的有代码（`twine check` 通过不代表有代码）：

```bash
python -m build
twine check dist/*
python -m zipfile -l dist/*.whl | grep llm_kernel/__init__.py
```

## 发布 / Release

版本号只有一处：`llm_kernel/__init__.py` 里的 `__version__`（`pyproject.toml` 用
`dynamic = ["version"]` 读它）。改完版本号、更新 `docs/changelog.md`，打 tag 并创建 Release，
`.github/workflows/publish.yml` 会构建、自检并上传 PyPI；`.github/workflows/docs.yml`
会把文档部署到 GitHub Pages。

认证二选一：仓库 secret `PYPI_API_TOKEN`（配了优先用），或 OIDC 可信发布
（在 PyPI 上登记 repository `lmliheng/llm_kernel`、workflow `publish.yml`、environment `pypi`）。

## 许可 / License

MIT，见 [LICENSE](LICENSE)。`llm_kernel/d2l.py` 是 d2l 项目自动生成的代码副本，
版权归原作者，遵循其原始许可。
