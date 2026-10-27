"""神经网络：激活函数、损失、指标、优化器和训练循环。

**torch 是可选依赖**：``import llm_kernel.nn`` 不需要装 torch，只有真正调用这些函数时才检查，
缺了会抛 :class:`llm_kernel.deps.MissingDependencyError` 并给出安装命令。
装法：``pip install "llm_kernel[nn]"``。

三个扩展入口（都不需要改库里的代码）：

1. **直接传函数**：``train_epoch(net, it, my_loss, my_updater)``——任何 callable 都收。
2. **注册名字**：``@llm_kernel.nn.METRICS.register("rmse")``，然后
   ``train_epoch(..., metrics=["accuracy", "rmse"])``。
3. **加一个模块**：在本目录新建文件（例如 ``schedulers.py``），在这里导出并加进 ``__all__``，
   别忘了给公开函数写 ``Examples:``。
"""

from . import activations, losses, metrics, optim, registry, training
from .activations import relu, softmax
from .losses import LOSSES, cross_entropy, cross_entropy_loss, squared_loss
from .metrics import METRICS, accuracy
from .optim import sgd
from .registry import Registry
from .training import Accumulator, train_epoch

__all__ = [
    "Accumulator",
    "LOSSES",
    "METRICS",
    "Registry",
    "accuracy",
    "activations",
    "cross_entropy",
    "cross_entropy_loss",
    "losses",
    "metrics",
    "optim",
    "registry",
    "relu",
    "sgd",
    "softmax",
    "squared_loss",
    "train_epoch",
    "training",
]
