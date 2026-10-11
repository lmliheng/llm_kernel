# llm_kernel

张量计算、机器学习、深度学习与模型训练推理的函数与最佳实践，边学边写、持续补充。

Reusable functions and best practices for tensor math, machine learning, deep learning and
model training / inference. Small, dependency-light and grown as the notes accumulate.

## 安装 / Installation

```bash
pip install llm_kernel
# 绘图模块需要 matplotlib / the plot module needs matplotlib
pip install "llm_kernel[plot]"
```

## 模块 / Modules

| 模块 Module | 内容 Contents |
| --- | --- |
| `llm_kernel.math` | 张量/数学计算，如正态分布 `normal` |
| `llm_kernel.performance` | 性能测试，如计时器 `Timer` |
| `llm_kernel.plot` | 二维可视化 `vector`（需要 matplotlib） |

## 用法 / Usage

### 正态分布 / Normal distribution

```python
import numpy as np
from llm_kernel.math import normal

normal(0.0, 0, 1)                       # 单点 0.3989422804014327
normal(np.array([-1.0, 0.0, 1.0]), 0, 1)  # 逐元素计算
```

`mu` 为均值，`sigma` 为标准差，必须为正实数。

### 计时 / Timing

```python
from llm_kernel.performance import Timer

t = Timer()
t.start()
# ... 需要测量的代码 ...
t.stop()

t.sum()      # 累计耗时
t.avg()      # 平均耗时
t.cumsum()   # 累计耗时序列
```

### 二维绘图 / 2D plotting

```python
from llm_kernel.plot import vector

vector([1, 2, 3], [1, 4, 9])  # x、y 长度必须相同
```

## 开发 / Development

```bash
git clone https://github.com/lmliheng/llm_kernel.git
cd llm_kernel
pip install -e ".[plot]"
```

## 发布 / Release

发布走 GitHub Actions（`.github/workflows/publish.yml`）：在 GitHub 上创建 Release（例如标签
`v0.1.0`），或在 Actions 页面手动触发该 workflow，即构建 sdist/wheel、安装 wheel 冒烟测试并上传 PyPI。

认证二选一，工作流两种都支持：

- 仓库 secret `PYPI_API_TOKEN`：配置了就优先用它上传；
- 不配置该 secret：自动走 OIDC 可信发布（Trusted Publishing），需要在 PyPI 上为该
  repository / workflow 注册发布者。

Publishing runs through GitHub Actions (`.github/workflows/publish.yml`): create a GitHub
Release (e.g. tag `v0.1.0`) or trigger the workflow manually, and it builds the sdist/wheel,
smoke-tests the wheel and uploads it to PyPI. It authenticates with the `PYPI_API_TOKEN`
repository secret when that secret exists, and otherwise falls back to OIDC trusted
publishing.
