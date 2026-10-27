"""数学与概率计算：只依赖 numpy。

新增一个函数的方式（固定配方，见 docs/guide/design.md）：

1. 在本目录新建或打开一个按主题命名的模块，例如 ``distributions.py``；
2. 函数用 ``snake_case``，被计算的对象放第一个参数，配置项关键字化并给默认值；
3. 在下面导出它，并加进 ``__all__``；
4. docstring 里写一条 ``Examples:``，``tests/test_docstrings.py`` 会真的跑它。
"""

from . import distributions
from .distributions import normal

__all__ = ["distributions", "normal"]
