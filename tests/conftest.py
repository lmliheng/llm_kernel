"""测试的公共设置：没有显示器也要能画图。"""

from __future__ import annotations

import os

# 必须在 import matplotlib 之前设置，两个测试环境（本地 / CI）都不弹窗口。
os.environ.setdefault("MPLBACKEND", "Agg")

try:  # matplotlib 是可选依赖，没装就跳过这一步
    import matplotlib

    matplotlib.use("Agg")
except ImportError:  # pragma: no cover - 取决于环境
    pass
