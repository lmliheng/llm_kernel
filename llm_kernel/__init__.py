"""llm_kernel —— 张量计算、机器学习、深度学习与模型训练的小工具集。

用法::

    import llm_kernel as lk

    lk.math.normal(0.0)            # 0.3989422804014327
    timer = lk.performance.Timer()  # 计时
    lk.nn.softmax(logits)          # 需要 torch

设计要点（详见 docs/guide/design.md）:

1. **一个入口**：所有能力都从 ``llm_kernel.<领域>`` 取用，领域是
   :mod:`llm_kernel.math`、:mod:`llm_kernel.nn`、:mod:`llm_kernel.performance`、
   :mod:`llm_kernel.plot`、:mod:`llm_kernel.deps`。
2. **惰性加载**：``import llm_kernel`` 不会导入 numpy / torch / matplotlib，
   只有真的访问某个子模块时才加载它（PEP 562 的模块级 ``__getattr__``），
   所以没装 torch 的机器也能 ``import llm_kernel``。
3. **可选依赖**：需要 torch 或 matplotlib 的函数在**调用时**才检查依赖，缺失时抛出
   :class:`llm_kernel.deps.MissingDependencyError`，消息里给出能直接复制的安装命令。
4. **向后兼容**：``math_``、``plot_``、``import_``、``NeuralNetwork`` 是 0.1.x 的旧名字，
   仍然可用（访问时提示一次 :class:`DeprecationWarning`），见 docs/guide/migration.md。
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.2.0"
"""库版本号，唯一来源；``pyproject.toml`` 用 ``dynamic = ["version"]`` 读它。"""

__all__ = ["math", "nn", "performance", "plot", "deps", "d2l", "__version__"]

#: 可以直接用 ``lk.xxx`` 访问的子模块，首次访问时才 import。
_SUBMODULES = ("math", "nn", "performance", "plot", "deps", "d2l")

#: 0.1.x 的旧子模块名，保留成兼容层，计划 1.0 移除（docs/guide/migration.md）。
_LEGACY_SUBMODULES = ("math_", "plot_", "import_", "NeuralNetwork")

_LAZY = _SUBMODULES + _LEGACY_SUBMODULES


def __getattr__(name: str):
    """按需导入子模块（PEP 562）。

    ``lk.nn`` 第一次被访问时才导入 ``llm_kernel.nn``，随后写回模块全局变量，
    再访问就不再经过这里；导入失败时子模块自己的报错会原样冒出来。
    """
    if name in _LAZY:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted({*globals(), *_LAZY})
