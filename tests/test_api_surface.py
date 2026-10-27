"""包层面的契约：惰性导入、旧名字兼容、构建产物里真的有代码。

这些用例专门盯着 0.1.x 出过的问题：
只想用 math 的人被 torch/matplotlib 拖累；改了目录结构以后 wheel 里一个模块都没有；
旧名字（math_ / plot_ / import_ / NeuralNetwork）突然失效。
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import pytest

import llm_kernel

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: 在子进程里屏蔽可选依赖，用来复现"没装 torch / matplotlib 的机器"。
BLOCK_OPTIONAL_DEPS = """
    import sys


    class BlockOptionalDeps:
        blocked = {
            "torch", "torchvision", "matplotlib", "matplotlib_inline",
            "pandas", "requests", "IPython",
        }

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in self.blocked:
                raise ImportError(f"测试里屏蔽了可选依赖：{name}")
            return None


    sys.meta_path.insert(0, BlockOptionalDeps())
    """


def run_python(*blocks: str) -> subprocess.CompletedProcess:
    """在干净的 Python 子进程里跑几段代码（每段各自去掉缩进后拼接）。"""
    code = "\n".join(textwrap.dedent(block) for block in blocks)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        # 固定用 utf-8 读子进程输出：Windows 上默认是 GBK，中文报错会解不出来
        encoding="utf-8",
        errors="replace",
    )


def test_importing_the_package_pulls_in_nothing():
    done = run_python(
        """
        import sys
        import llm_kernel
        print([m for m in ("numpy", "torch", "matplotlib") if m in sys.modules])
        """
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_submodules_import_without_optional_dependencies():
    done = run_python(
        BLOCK_OPTIONAL_DEPS,
        """
        import llm_kernel
        import llm_kernel.math
        import llm_kernel.nn
        import llm_kernel.performance
        import llm_kernel.plot
        import llm_kernel.deps
        print("ok", llm_kernel.math.normal(0.0) > 0, len(llm_kernel.nn.__all__) > 0)
        """,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("ok True True")


def test_calling_nn_without_torch_prints_an_install_command():
    done = run_python(
        BLOCK_OPTIONAL_DEPS,
        """
        import llm_kernel
        try:
            llm_kernel.nn.relu([1, 2])
        except ImportError as exc:
            print(type(exc).__name__, "|", exc)
        """,
    )
    assert done.returncode == 0, done.stderr
    assert "MissingDependencyError" in done.stdout
    assert 'pip install "llm_kernel[nn]"' in done.stdout


def test_calling_plot_without_matplotlib_prints_an_install_command():
    done = run_python(
        BLOCK_OPTIONAL_DEPS,
        """
        import llm_kernel
        try:
            llm_kernel.plot.curve([1, 2], [3, 4], show=False)
        except ImportError as exc:
            print(type(exc).__name__, "|", exc)
        """,
    )
    assert done.returncode == 0, done.stderr
    assert 'pip install "llm_kernel[plot]"' in done.stdout


def test_legacy_names_still_work_and_warn_once_each():
    done = run_python(
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from llm_kernel.math_ import normal
            from llm_kernel.plot_ import vector
            from llm_kernel.import_ import isImport
            from llm_kernel.NeuralNetwork import softmax, relu, sgd, accuracy, cross_entropy_loss
            from llm_kernel.performance import Timer, timer
            from llm_kernel.performance.Timer import Timer as TimerFromModule

        print([
            normal.__name__, vector.__name__, isImport.__name__, softmax.__name__,
            relu.__name__, sgd.__name__, accuracy.__name__, cross_entropy_loss.__name__,
            Timer.__name__, timer.__name__, TimerFromModule.__name__,
        ])
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        print(len(messages), all("已更名" in m for m in messages))
        """
    )
    assert done.returncode == 0, done.stderr
    first, second = done.stdout.strip().splitlines()
    # cross_entropy_loss 是 cross_entropy 的别名，__name__ 就是新名字
    assert "softmax" in first and "cross_entropy" in first
    count, all_warned = second.split()
    assert int(count) >= 4, done.stdout
    assert all_warned == "True"


def test_public_surface_is_discoverable():
    for name in ("math", "nn", "performance", "plot", "deps", "d2l"):
        assert name in dir(llm_kernel)
    for legacy in ("math_", "plot_", "import_", "NeuralNetwork"):
        assert legacy in dir(llm_kernel)
    assert llm_kernel.__version__.count(".") == 2


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        llm_kernel.根本没这个子模块  # noqa: B018


def test_installed_package_ships_the_code():
    """构建产物是空包时（0.1.x 踩过的坑）这条会失败。"""
    package_dir = pathlib.Path(llm_kernel.__file__).parent
    for relative in (
        "math/distributions.py",
        "nn/training.py",
        "performance/timing.py",
        "plot/curves.py",
        "deps/requirements.py",
        "py.typed",
    ):
        assert (package_dir / relative).is_file(), f"包内缺少 {relative}"


def test_version_has_a_single_source():
    tomllib = pytest.importorskip("tomllib")  # 3.11+ 才有
    config_path = REPO_ROOT / "pyproject.toml"
    if not config_path.is_file():
        pytest.skip("不在源码仓库里")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "version" in config["project"]["dynamic"]
    assert "version" not in config["project"]
    assert config["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "llm_kernel.__version__"


def test_packaging_config_matches_the_real_layout():
    """pyproject 里说包在 `llm_kernel/` 下，那就得真的有这个目录（0.1.x 的包结构 bug）。"""
    config_path = REPO_ROOT / "pyproject.toml"
    if not config_path.is_file():
        pytest.skip("不在源码仓库里")
    assert (REPO_ROOT / "llm_kernel" / "__init__.py").is_file()
    assert not (REPO_ROOT / "__init__.py").exists()
