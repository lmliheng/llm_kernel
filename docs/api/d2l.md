# `llm_kernel.d2l` — 内置的 d2l 工具库副本

```python
from llm_kernel import d2l

d2l.Timer
d2l.load_data_fashion_mnist(256)
d2l.train_ch3(...)
```

## 这是什么

《动手学深度学习》（d2l）配套代码里的工具库，由 `d2lbook build lib` 自动生成后收进本库，
方便在 notebook 里少写几行样板代码。它**原样保留**（除了搬进包目录，没有改动内容），
所以别指望这里的名字和 `llm_kernel.nn` 完全一致——d2l 有自己的历史写法。

## 和库其它部分的关系

| | `llm_kernel.d2l` | `llm_kernel.nn` / 其它子模块 |
| --- | --- | --- |
| 来源 | 第三方（d2l 书，自动生成） | 本库自己写的 |
| 依赖 | torch、torchvision、pandas、requests、ipython、matplotlib | 各自按需 |
| 风格 | 教学示例，名字按书里的章节排列 | 统一命名与错误约定的工具箱 |
| 导入 | `from llm_kernel import d2l` | `import llm_kernel.nn as nn` |

**建议**：学习时随书用 `d2l`，自己写项目时用 `llm_kernel.nn` 等模块。
两者共存，不互相依赖。

## 依赖

```bash
pip install "llm_kernel[d2l]"
```

`import llm_kernel.d2l` 需要上面这些包都在位——它内部是直接 import 的（第三方代码保持原样）。
好在这不影响别人：`import llm_kernel` 本身不会导入它。

## 版权

`d2l.py` 是 d2l 项目自动生成的代码副本，版权归 d2l 作者所有，遵循其原始许可；
本库的 MIT 许可只覆盖自己写的部分。文件头保留了自动生成的标记，请勿手工编辑该文件
（要改就在上游改，或在自己代码里封装）。
