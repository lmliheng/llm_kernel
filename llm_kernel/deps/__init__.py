"""依赖检查：探测可选依赖，并在缺失时给出能直接照抄的安装命令。

这个子包自己只依赖标准库，所以任何环境里都能 ``import llm_kernel.deps``。
"""

from .requirements import (
    MissingDependencyError,
    is_installed,
    require,
    require_all,
    version_of,
)

__all__ = [
    "MissingDependencyError",
    "is_installed",
    "require",
    "require_all",
    "version_of",
]
