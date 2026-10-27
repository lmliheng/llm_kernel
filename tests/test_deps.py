"""`llm_kernel.deps` 的测试：缺依赖时的报错必须是人话。"""

from __future__ import annotations

import types

import pytest

from llm_kernel import deps


def test_is_installed():
    assert deps.is_installed("json") is True
    assert deps.is_installed("绝对不存在的模块名_xyz") is False


def test_version_of_known_and_unknown():
    assert deps.version_of("绝对不存在的包名_xyz") is None
    assert deps.version_of("json") is None  # 标准库不是发行包
    assert isinstance(deps.version_of("numpy"), str)


def test_require_returns_module():
    assert deps.require("json").__name__ == "json"


def test_require_all_returns_versions_in_order():
    got = deps.require_all(["json", "math"])
    assert list(got) == ["json", "math"]
    assert got == {"json": None, "math": None}


def test_missing_dependency_error_is_import_error():
    with pytest.raises(deps.MissingDependencyError) as info:
        deps.require("绝对不存在的模块名_xyz")
    assert "pip install" in str(info.value)
    assert isinstance(info.value, ImportError)  # 老的 except ImportError 还能接住


def test_require_all_stops_at_first_missing():
    with pytest.raises(deps.MissingDependencyError):
        deps.require_all(["json", "绝对不存在的模块名_xyz"])


def test_known_dependency_hint_points_at_extra(monkeypatch):
    """torch 缺失时要提示 `pip install "llm_kernel[nn]"`，而不是让使用者自己去猜。"""

    def boom(name):
        raise ImportError("模拟：这里没有装")

    monkeypatch.setattr(deps.requirements, "importlib", types.SimpleNamespace(import_module=boom))
    with pytest.raises(deps.MissingDependencyError) as info:
        deps.require("torch")
    assert 'pip install "llm_kernel[nn]"' in str(info.value)


def test_unknown_dependency_uses_extra_argument(monkeypatch):
    def boom(name):
        raise ImportError("模拟：这里没有装")

    monkeypatch.setattr(deps.requirements, "importlib", types.SimpleNamespace(import_module=boom))
    with pytest.raises(deps.MissingDependencyError) as info:
        deps.require("some_unknown_pkg", extra="plot")
    assert 'pip install "llm_kernel[plot]"' in str(info.value)
