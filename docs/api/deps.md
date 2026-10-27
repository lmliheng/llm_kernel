# `llm_kernel.deps` — 依赖检查

只依赖标准库，任何环境里都能导入。库内部就是靠它把"缺可选依赖"变成人话。

```python
from llm_kernel.deps import is_installed, require, require_all, version_of

is_installed("torch")      # 只查不导，没有副作用
version_of("numpy")        # '2.5.3'；标准库或没装的包返回 None
require("torch")           # 没装就抛 MissingDependencyError，消息里带安装命令
require_all(["torch"])     # {'torch': '2.14.1+cpu'}
```

`MissingDependencyError` 继承自 `ImportError`：老代码的 `except ImportError` 依然能接住，
想精确处理就 `except llm_kernel.deps.MissingDependencyError`。

::: llm_kernel.deps.requirements
    options:
      members_order: source
