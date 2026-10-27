# `llm_kernel.nn` — 神经网络

torch 是**可选依赖**：`import llm_kernel.nn` 不需要装 torch，只有在调用函数时才检查。
装法：`pip install "llm_kernel[nn]"`。

三条约定贯穿整个模块：

1. **损失** `loss(y_hat, y) -> torch.Tensor`，默认返回**逐样本**的向量（由调用方决定怎么归约）。
2. **指标** `metric(y_hat, y) -> float`，返回本批的比例，由 `train_epoch()` 按样本数加权汇总。
3. **优化器** `optimizer(params, lr, batch_size) -> None`，原地更新参数。

```python
from llm_kernel.nn import sgd, squared_loss, train_epoch

w = torch.zeros((2, 1), requires_grad=True)

def updater(batch_size):
    sgd([w], lr=0.05, batch_size=batch_size)

train_epoch(lambda X: X @ w, data, squared_loss, updater, metrics=[])   # {'loss': …}
```

## 激活函数

::: llm_kernel.nn.activations
    options:
      members_order: source

## 损失函数

注册表 `LOSSES` 里已经有 `cross_entropy` 和 `squared_loss`；调用时既能写名字也能直接传函数。
自定义损失只要符合 `loss(y_hat, y) -> torch.Tensor` 就能直接用，不必注册。

::: llm_kernel.nn.losses
    options:
      members_order: source

## 指标

注册表 `METRICS` 里已经有 `accuracy`。

::: llm_kernel.nn.metrics
    options:
      members_order: source

## 优化器

::: llm_kernel.nn.optim
    options:
      members_order: source

## 训练循环

::: llm_kernel.nn.training
    options:
      members_order: source

## 注册表

::: llm_kernel.nn.registry
    options:
      members_order: source
