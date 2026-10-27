"""可选依赖的探测与友好报错。

约定：库内部**在函数里**调用 :func:`require`，而不是在模块顶层 ``import torch``。
这样 ``import llm_kernel.nn`` 在没有 torch 的机器上也能成功，报错推迟到真正调用函数时，
而且报错里带着可以直接复制的安装命令。
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from importlib.util import find_spec
from types import ModuleType
from typing import Iterable

__all__ = [
    "MissingDependencyError",
    "is_installed",
    "require",
    "require_all",
    "version_of",
]

#: 常见可选依赖 → 推荐的安装命令，``require`` 报错时用它拼提示。
_INSTALL_HINT = {
    "torch": 'pip install "llm_kernel[nn]"',
    "matplotlib": 'pip install "llm_kernel[plot]"',
    "torchvision": 'pip install "llm_kernel[d2l]"',
    "pandas": 'pip install "llm_kernel[d2l]"',
    "requests": 'pip install "llm_kernel[d2l]"',
    "IPython": 'pip install "llm_kernel[d2l]"',
    "matplotlib_inline": 'pip install "llm_kernel[d2l]"',
}


class MissingDependencyError(ImportError):
    """缺少可选依赖。

    继承自 :class:`ImportError`，所以既可以被 ``except ImportError`` 接住，
    也可以单独 ``except MissingDependencyError`` 只处理这种情况。
    """


def is_installed(name: str) -> bool:
    """模块 ``name`` 能不能被导入（只查不导，因此没有副作用）。

    Args:
        name (str): 模块名，例如 ``"torch"``。

    Returns:
        bool: 找得到返回 ``True``。

    Examples:
        >>> from llm_kernel.deps import is_installed
        >>> is_installed("这是一个不存在的模块名")
        False
    """
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        # 父模块不存在、名字非法等情况都算“没装”
        return False


def version_of(name: str) -> str | None:
    """已安装发行包的版本号；不是发行包（如标准库）或没装时返回 ``None``。

    Args:
        name (str): **发行包名**，不是导入名。例如装的是 ``Pillow`` 而导入名是 ``PIL``，
            就要传 ``"Pillow"``。

    Returns:
        str | None: 例如 ``"2.5.3"``；标准库模块返回 ``None``。

    Examples:
        >>> from llm_kernel.deps import version_of
        >>> version_of("llm_kernel") is None or isinstance(version_of("llm_kernel"), str)
        True
    """
    try:
        return _dist_version(name)
    except PackageNotFoundError:
        return None


def require(name: str, *, extra: str | None = None) -> ModuleType:
    """导入并返回模块 ``name``；缺失时抛出带安装命令的 :class:`MissingDependencyError`。

    Args:
        name (str): 模块名，例如 ``"torch"``。
        extra (str | None): 兜底提示里用的 extra 名，例如 ``"nn"`` → ``pip install "llm_kernel[nn]"``。
            常见依赖不用传，库里已经有默认提示。

    Returns:
        types.ModuleType: 导入好的模块对象。

    Raises:
        MissingDependencyError: 模块不可用（没装、装坏了、或它自己的依赖缺失）。

    Examples:
        >>> from llm_kernel.deps import require
        >>> require("json").__name__
        'json'
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        hint = _INSTALL_HINT.get(name)
        if hint is None:
            hint = f'pip install "llm_kernel[{extra}]"' if extra else f"pip install {name}"
        raise MissingDependencyError(
            f"缺少可选依赖 {name!r}，无法使用这个功能；安装方式：{hint}"
        ) from exc


def require_all(names: Iterable[str]) -> dict:
    """批量检查依赖，返回 ``{模块名: 版本号或 None}``。

    Args:
        names (Iterable[str]): 模块名集合。

    Returns:
        dict: 模块名 → 版本号（不是发行包时是 ``None``），顺序与传入一致。

    Raises:
        MissingDependencyError: 只要有一个模块不可用就抛出（保留异常链，方便定位是谁没装）。

    Examples:
        >>> from llm_kernel.deps import require_all
        >>> require_all(["json", "math"])  # 标准库没有版本号
        {'json': None, 'math': None}
    """
    versions = {}
    for name in names:
        require(name)
        versions[name] = version_of(name)
    return versions
