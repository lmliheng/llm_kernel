"""`llm_kernel.plot` 的测试。没有 matplotlib 就整体跳过。"""

from __future__ import annotations

import warnings

import pytest

matplotlib = pytest.importorskip("matplotlib")

from matplotlib import pyplot as plt  # noqa: E402

from llm_kernel.plot import curve, vector  # noqa: E402


def test_returns_figure_and_axes_with_labels():
    fig, ax = curve(
        [1, 2, 3],
        [1, 4, 9],
        label="平方",
        xlabel="x",
        ylabel="y",
        title="标题",
        show=False,
    )
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_title() == "标题"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["平方"]
    plt.close(fig)


def test_no_label_means_no_legend_and_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 0.1.x 在这里会报 legend 警告
        fig, ax = curve([1, 2], [3, 4], show=False)
    assert ax.get_legend() is None
    plt.close(fig)


def test_length_mismatch_is_a_value_error():
    with pytest.raises(ValueError, match="长度必须相同"):
        curve([1, 2], [1], show=False)


def test_non_sequence_is_a_type_error():
    with pytest.raises(TypeError, match="可求长度的序列"):
        curve(1, 2, show=False)


def test_can_stack_curves_on_existing_axes():
    fig, ax = curve([0, 1], [0, 1], label="一条", show=False)
    fig2, ax2 = curve([0, 1], [1, 0], label="两条", ax=ax, show=False)
    assert fig2 is fig and ax2 is ax
    assert len(ax.get_lines()) == 2
    assert len(ax.get_legend().get_texts()) == 2
    plt.close(fig)


def test_show_calls_plt_show(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: calls.append(1))
    fig, _ = curve([0, 1], [0, 1])
    assert calls == [1]
    plt.close(fig)


def test_plot_kwargs_are_forwarded():
    fig, ax = curve([0, 1], [0, 1], show=False, color="r", linestyle="--")
    assert ax.get_lines()[0].get_color() == "r"
    assert ax.get_lines()[0].get_linestyle() == "--"
    plt.close(fig)


def test_old_vector_name_keeps_the_old_style():
    fig, ax = curve([0, 1], [0, 1], show=False)  # 新名字：默认颜色
    fig2, ax2 = vector([0, 1], [0, 1], show=False)  # 旧名字：蓝色粗线
    assert ax.get_lines()[0].get_color() != "b"
    assert ax2.get_lines()[0].get_color() == "b"
    assert ax2.get_lines()[0].get_linewidth() == 2
    plt.close(fig)
    plt.close(fig2)
