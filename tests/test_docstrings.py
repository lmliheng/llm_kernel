"""docstring 里的 ``Examples:`` 必须真的能跑。

文档和代码分开腐烂是很容易发生的事：改了实现忘了改示例，读者照着抄就报错。
这些用例把每个公开模块的示例都当测试跑一遍。
"""

from __future__ import annotations

import doctest
import importlib

import pytest

OPTIONS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS

DOCUMENTED_MODULES = [
    "llm_kernel.math.distributions",
    "llm_kernel.deps.requirements",
    "llm_kernel.performance.timing",
    "llm_kernel.plot.curves",
    "llm_kernel.nn.registry",
    "llm_kernel.nn.activations",
    "llm_kernel.nn.losses",
    "llm_kernel.nn.metrics",
    "llm_kernel.nn.optim",
    "llm_kernel.nn.training",
]


@pytest.mark.parametrize("module_name", DOCUMENTED_MODULES)
def test_docstring_examples(module_name):
    if module_name.startswith("llm_kernel.nn"):
        pytest.importorskip("torch")
    if module_name.startswith("llm_kernel.plot"):
        pytest.importorskip("matplotlib")

    module = importlib.import_module(module_name)
    result = doctest.testmod(module, optionflags=OPTIONS, verbose=False)
    assert result.attempted > 0, f"{module_name} 的 docstring 里一条可运行示例都没有"
    assert result.failed == 0, f"{module_name} 有 {result.failed} 条示例没通过"
