"""`llm_kernel.nn` 的行为测试。没有 torch 就整体跳过（CI 里有一个专门装 torch 的任务）。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llm_kernel.nn import (  # noqa: E402
    LOSSES,
    METRICS,
    Accumulator,
    Registry,
    accuracy,
    cross_entropy,
    cross_entropy_loss,
    relu,
    sgd,
    softmax,
    squared_loss,
    train_epoch,
)


# --- 激活函数 ---------------------------------------------------------------


def test_relu_matches_torch_and_keeps_shape():
    x = torch.tensor([[-1.0, 0.0], [2.0, -3.0]])
    assert torch.equal(relu(x), torch.nn.functional.relu(x))
    assert relu(x).shape == x.shape


def test_softmax_matches_torch_and_is_stable():
    x = torch.tensor([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
    got = softmax(x)
    assert torch.allclose(got, torch.nn.functional.softmax(x, dim=-1))
    assert torch.allclose(got.sum(dim=-1), torch.ones(2))


def test_softmax_respects_dim():
    x = torch.randn(2, 3, 4)
    assert torch.allclose(softmax(x, dim=0).sum(dim=0), torch.ones(3, 4))


# --- 损失函数 ---------------------------------------------------------------


def test_cross_entropy_values_and_reductions():
    p = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    y = torch.tensor([0, 1])
    per_sample = cross_entropy(p, y)
    assert per_sample.shape == (2,)
    assert torch.allclose(per_sample, torch.tensor([-torch.log(torch.tensor(0.9)), -torch.log(torch.tensor(0.8))]))
    assert torch.allclose(cross_entropy(p, y, reduction="mean"), per_sample.mean())
    assert torch.allclose(cross_entropy(p, y, reduction="sum"), per_sample.sum())


def test_cross_entropy_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="reduction 只能是"):
        cross_entropy(torch.tensor([[1.0, 0.0]]), torch.tensor([0]), reduction="avg")


def test_old_loss_name_is_the_same_function():
    assert cross_entropy_loss is cross_entropy


def test_squared_loss_definition():
    got = squared_loss(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0]))
    assert torch.allclose(got, torch.tensor([0.0, 2.0]))


def test_losses_registry_has_the_builtin_names():
    assert "cross_entropy" in LOSSES and "squared_loss" in LOSSES


# --- 指标 -------------------------------------------------------------------


def test_accuracy_from_logits_and_from_labels():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    y = torch.tensor([1, 1, 1])
    assert accuracy(logits, y) == pytest.approx(2 / 3)
    assert accuracy(torch.tensor([1, 0, 1]), y) == pytest.approx(2 / 3)


def test_accuracy_accepts_column_vector_labels():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    assert accuracy(logits, torch.tensor([[1], [0]])) == pytest.approx(1.0)


def test_accuracy_rejects_empty_labels():
    with pytest.raises(ValueError, match="不能为空"):
        accuracy(torch.tensor([[1.0, 0.0]]), torch.tensor([], dtype=torch.long))


# --- 优化器 -----------------------------------------------------------------


def test_sgd_does_one_step_and_clears_gradient():
    w = torch.tensor([1.0], requires_grad=True)
    (w * 2).sum().backward()
    sgd([w], lr=0.1, batch_size=2)
    assert w.item() == pytest.approx(0.9)
    assert w.grad.item() == 0.0


def test_sgd_validates_arguments():
    w = torch.tensor([1.0], requires_grad=True)
    (w * 2).sum().backward()
    with pytest.raises(ValueError, match="lr 必须为正数"):
        sgd([w], lr=0, batch_size=1)
    with pytest.raises(ValueError, match="batch_size 必须为正整数"):
        sgd([w], lr=0.1, batch_size=0)


def test_sgd_without_gradient_explains_itself():
    with pytest.raises(ValueError, match="先对损失调用 backward"):
        sgd([torch.tensor([1.0], requires_grad=True)], lr=0.1, batch_size=1)


# --- 累加器 -----------------------------------------------------------------


def test_accumulator_behaviour():
    acc = Accumulator(2)
    acc.add(1.0, 3.0)
    acc.add(2.0, 5.0)
    assert (acc[0], acc[1]) == (3.0, 8.0)
    assert len(acc) == 2 and list(acc) == [3.0, 8.0] and acc.values == [3.0, 8.0]
    acc.values.append(9.0)  # 是副本，改它不影响内部
    assert acc.values == [3.0, 8.0]
    acc.reset()
    assert acc.values == [0.0, 0.0]


def test_accumulator_validates_arity():
    acc = Accumulator(2)
    with pytest.raises(ValueError, match="收到 1 个值"):
        acc.add(1.0)
    with pytest.raises(ValueError, match="n 必须为正整数"):
        Accumulator(0)


# --- 训练循环 ---------------------------------------------------------------


def _linear_regression_data(num=64, noise=0.0):
    generator = torch.Generator().manual_seed(0)
    X = torch.randn(num, 2, generator=generator)
    w_true = torch.tensor([[3.0], [-2.0]])
    b_true = 1.5
    y = X @ w_true + b_true
    if noise:
        y = y + noise * torch.randn(num, 1, generator=generator)
    return X, y


def _batches(X, y, batch_size):
    return [(X[i : i + batch_size], y[i : i + batch_size]) for i in range(0, len(X), batch_size)]


def test_train_epoch_with_custom_updater_learns_and_reports():
    X, y = _linear_regression_data()
    data = _batches(X, y, 16)
    w = torch.zeros((2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    params = [w, b]

    def updater(batch_size):
        sgd(params, lr=0.05, batch_size=batch_size)

    def net(X):
        return X @ w + b

    history = [train_epoch(net, data, squared_loss, updater, metrics=[])["loss"] for _ in range(30)]
    assert history[0] > history[-1]
    assert history[-1] < 1e-2

    with_torch_optimizer = torch.optim.SGD(params, lr=0.05)
    result = train_epoch(net, data, squared_loss, with_torch_optimizer, metrics=[])
    assert set(result) == {"loss"}


def test_train_epoch_accepts_registered_and_custom_metrics():
    X, y = _linear_regression_data()

    @METRICS.register("always_one")
    def always_one(y_hat, y):
        return 1.0

    def half(y_hat, y):
        return 0.5

    w = torch.zeros((2, 1), requires_grad=True)

    def updater(batch_size):
        sgd([w], lr=0.01, batch_size=batch_size)

    result = train_epoch(
        lambda X: X @ w,
        _batches(X, y, 32),
        squared_loss,
        updater,
        metrics=["always_one", half],
    )
    assert result["always_one"] == pytest.approx(1.0)
    assert result["half"] == pytest.approx(0.5)
    assert "loss" in result


def test_train_epoch_accepts_torch_nn_loss_and_optimizer():
    X, y = _linear_regression_data()
    labels = (X[:, 0] > 0).long()  # 二分类玩具任务
    layer = torch.nn.Linear(2, 2)
    net = torch.nn.Sequential(layer)
    criterion = torch.nn.CrossEntropyLoss()  # 已经归约成标量
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    first = train_epoch(net, _batches(X, labels, 32), criterion, optimizer, metrics=["accuracy"])
    for _ in range(40):
        last = train_epoch(net, _batches(X, labels, 32), criterion, optimizer, metrics=["accuracy"])

    assert set(first) == {"loss", "accuracy"}
    assert last["loss"] < first["loss"]
    assert last["accuracy"] > 0.9


def test_train_epoch_loss_accepts_registered_name():
    X, y = _linear_regression_data(num=16)
    w = torch.zeros((2, 1), requires_grad=True)

    def updater(batch_size):
        sgd([w], lr=0.01, batch_size=batch_size)

    result = train_epoch(lambda X: X @ w, _batches(X, y, 8), "squared_loss", updater, metrics=[])
    assert "loss" in result


def test_train_epoch_errors_are_clear():
    with pytest.raises(ValueError, match="一个样本都没产出"):
        train_epoch(lambda X: X, [], squared_loss, lambda n: None, metrics=[])
    with pytest.raises(KeyError, match="没有注册名为"):
        train_epoch(lambda X: X, [], squared_loss, lambda n: None, metrics=["并不存在的指标"])
    with pytest.raises(TypeError, match="只能是名字或可调用对象"):
        train_epoch(lambda X: X, [], squared_loss, lambda n: None, metrics=[123])


# --- 注册表 -----------------------------------------------------------------


def test_registry_rejects_duplicate_names():
    registry = Registry("指标")
    registry.register("a", lambda *args: 0.0)
    with pytest.raises(ValueError, match="已经注册过了"):
        registry.register("a", lambda *args: 0.0)


def test_registry_get_unknown_lists_available_names():
    registry = Registry("指标")
    registry.register("a", lambda *args: 0.0)
    with pytest.raises(KeyError, match="已注册：\\['a'\\]"):
        registry.get("b")


def test_metrics_registry_has_accuracy():
    assert "accuracy" in METRICS
    assert METRICS.get("accuracy") is accuracy
