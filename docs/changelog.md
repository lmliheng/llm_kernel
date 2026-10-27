# 变更日志

## 0.2.0 — 未发布

一次把结构、命名、依赖、测试、文档全部补齐的重构。**旧名字全部保留**（见迁移指南）。

### 新增

- `llm_kernel.nn.train_epoch()`：训练一个 epoch，返回 `{"loss": …, <指标名>: …}` 字典，
  加指标不用改调用方代码。
- `llm_kernel.nn.LOSSES` / `METRICS` 注册表与 `Registry` 类：用装饰器把自己的损失/指标接进来，
  也可以直接传函数（两条路都通）。
- `llm_kernel.nn.softmax()`：数值稳定（先减最大值）。
- `llm_kernel.deps` 子包：`require` / `require_all` / `is_installed` / `version_of` /
  `MissingDependencyError`。
- `llm_kernel.plot.curve()`：替代 `vector`，返回 `(fig, ax)`，支持 `ax=`（叠加曲线）、
  `show=`、`xlabel/ylabel/title/label`、透传 `**plot_kwargs`。
- `Timer`：`with Timer() as t`、`last`、`count`、`elapsed`、`reset()`；
  `@timer(verbose=False, precision=…)`。
- `Accumulator`：`len()`、迭代、`values`、`reset()`。
- 工程化：`py.typed`（PEP 561）、LICENSE、classifiers、`docs/` 文档站、
  测试套件、CI（测试 / 文档 / 发布三个工作流）。

### 变更

- **目录结构**：扁平布局（仓库根目录直接就是包）改为 `llm_kernel/` 包目录。
  修好之后 `pyproject.toml` 里的 `packages.find` 才真的能找到包。
- **命名**：`NeuralNetwork` → `nn`；`import_` → `deps`；`math_` → `math`；`plot_` → `plot`；
  `performance.Timer`（模块）→ `performance.timing`。
- **版本号单一来源**：`llm_kernel/__init__.py` 的 `__version__`，
  `pyproject.toml` 用 `dynamic = ["version"]`；发布工作流会校验 tag 与它一致。
- 事件函数返回类型：`Timer.cumsum()`、`normal()` 等不再返回 numpy 标量。

### 修复

- `NeuralNetwork/train.py` 的顶层导入 `from accuracy import accuracy`：一 import 就
  `ModuleNotFoundError`，导致那里的 `Accumulator` 和训练函数在 0.1.x 里实际不可用。
- `plot.vector()` 没给 `label` 却调用 `plt.legend()`（matplotlib 警告），且没有返回值。
- `Timer.avg()` 空列表 `ZeroDivisionError`；`stop()` 前没 `start()` 时 `AttributeError`；
  改用单调时钟 `time.perf_counter()`。
- `normal()`：`list` 输入报 `TypeError`、`mu/sigma` 的类型标注写成 `int`、
  参数错误抛裸 `Exception`、标量输入返回 numpy 标量。
- `accuracy()`：空批次除零；用了已废弃的 `axis=` 别名。
- `isImport()` 往 stdout 打印、抛裸 `Exception`。
- 误提交进版本库的 `__pycache__/*.pyc` 已移出（`.gitignore` 补齐）。
- 构建产物空包问题：0.1.0 之后的 main 分支构建出的 wheel 里没有任何模块
  （`packages.find` 找不到包），发布前一定会踩到。

### 说明

- 仓库里曾经出现过 `__version__ = "0.1.1"`，但**从未发布到 PyPI**；
  0.2.0 是本库第一个带测试、文档和 CI 的版本。

## 0.1.0 — 2026-10-10

首个 PyPI 版本（`llm_kernel`，作者 lmliheng）。

- `llm_kernel.math.normal()`：正态分布概率密度。
- `llm_kernel.performance.Timer`：计时器。
- `llm_kernel.plot.vector()`：二维折线图（需要 matplotlib）。
- 依赖：`numpy>=1.20`，`matplotlib` 走 `plot` extra。
- 发布：`.github/workflows/publish.yml`，支持 `PYPI_API_TOKEN` 或 OIDC 可信发布。
