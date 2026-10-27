"""`llm_kernel.math` 的行为测试，重点覆盖 0.1.x 里踩过的坑。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from llm_kernel.math import normal

SQRT_2PI = math.sqrt(2 * math.pi)


def test_scalar_in_scalar_out():
    value = normal(0.0)
    assert isinstance(value, float)  # 直接能 json 序列化（0.1.x 返回的是 numpy 标量）
    assert value == pytest.approx(1 / SQRT_2PI)


def test_defaults_are_standard_normal():
    assert normal(0.0) == pytest.approx(normal(0.0, mu=0.0, sigma=1.0))


@pytest.mark.parametrize("x", [[-1.0, 0.0, 1.0], (-1.0, 0.0, 1.0), np.array([-1.0, 0.0, 1.0])])
def test_array_like_input_returns_ndarray(x):
    got = normal(x)
    assert isinstance(got, np.ndarray)
    np.testing.assert_allclose(got, [0.24197072, 0.39894228, 0.24197072])


def test_torch_tensor_input_works_without_importing_torch_here():
    torch = pytest.importorskip("torch")
    got = normal(torch.tensor([0.0, 1.0]))
    assert isinstance(got, np.ndarray)
    np.testing.assert_allclose(got, normal(np.array([0.0, 1.0])))


def test_matches_analytic_formula_on_grid():
    xs = np.linspace(-3.0, 3.0, 25)
    expect = np.exp(-(((xs - 0.5) / 2.0) ** 2) / 2) / (2.0 * SQRT_2PI)
    np.testing.assert_allclose(normal(xs, mu=0.5, sigma=2.0), expect, rtol=1e-12)


def test_shape_preserved_for_2d_input():
    xs = np.zeros((2, 3))
    assert normal(xs).shape == (2, 3)


def test_non_positive_sigma_raises_value_error():
    with pytest.raises(ValueError, match="sigma 必须为正数"):
        normal(0.0, sigma=0)


def test_bad_types_raise_type_error_with_clear_message():
    with pytest.raises(TypeError, match="sigma 必须是实数"):
        normal(0.0, sigma="1")
    with pytest.raises(TypeError, match="mu 必须是实数"):
        normal(0.0, mu=[0])
    with pytest.raises(TypeError, match="x 必须是标量"):
        normal({"a": 1})


def test_bool_is_not_treated_as_number():
    with pytest.raises(TypeError, match="mu 必须是实数"):
        normal(0.0, mu=True)
