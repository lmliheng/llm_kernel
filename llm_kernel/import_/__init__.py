"""兼容层：``llm_kernel.import_`` 已更名为 :mod:`llm_kernel.deps`。

除了转发新名字，还保留了 0.1.x 的 ``isImport()``。这里只做转发 + 提醒一次，1.0 版本删除。
"""

import warnings

from .. import deps as _current
from ..deps import *  # noqa: F401,F403
from ..deps import __all__ as _new_all

__all__ = [*_new_all, "isImport"]


def isImport(modules: list) -> None:
    """0.1.x 的旧函数：逐个检查依赖、打印版本，缺一个就抛异常。

    .. deprecated::
        请改用 :func:`llm_kernel.deps.require_all`——它返回 ``{模块名: 版本号}`` 而不打印，
        库函数不该往 stdout 打字。

    Examples:
        >>> from llm_kernel.import_ import isImport
        >>> isImport(["json"])
        json版本： None
    """
    for name, version in _current.require_all(modules).items():
        print(f"{name}版本：", version)


def __getattr__(name: str):
    """新模块以后新增的名字，这里也能自动取到。"""
    return getattr(_current, name)


warnings.warn(
    "llm_kernel.import_ 已更名为 llm_kernel.deps，请改用新名字（见 docs/guide/migration.md）",
    DeprecationWarning,
    stacklevel=2,
)
