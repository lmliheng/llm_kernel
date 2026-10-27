"""一个 20 行的小注册表：让使用者能把自己写的损失/指标接进库里。

为什么不做插件系统：库要"好懂"就不能让使用者去读框架代码。一个字典 + 装饰器，
看一眼就知道怎么回事；而且函数本身照旧可以直接当参数传（鸭子类型），
注册只是让"按名字用"变成可能。
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["Registry"]


class Registry:
    """名字 → 函数 的注册表，支持三种写法。

    Examples:
        >>> from llm_kernel.nn.registry import Registry
        >>> METRICS = Registry("指标")     # 这里用局部实例演示，不动库里的全局注册表
        >>> @METRICS.register("hit_rate")  # 自己起名字
        ... def hit_rate(y_hat, y):
        ...     return 1.0
        >>> METRICS.get("hit_rate")(None, None)
        1.0
        >>> @METRICS.register               # 用函数自己的名字当键
        ... def top1_accuracy(y_hat, y):
        ...     return 1.0
        >>> METRICS.names()
        ['hit_rate', 'top1_accuracy']
        >>> "top1_accuracy" in METRICS, len(METRICS)
        (True, 2)
        >>> METRICS.get("没有这个名字")
        Traceback (most recent call last):
            ...
        KeyError: "没有注册名为 '没有这个名字' 的指标；已注册：['hit_rate', 'top1_accuracy']"
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict = {}

    def register(
        self,
        name: Callable | str | None = None,
        fn: Callable | None = None,
    ) -> Callable:
        """登记一个函数；``name`` 省略时用函数自己的 ``__name__``。

        Args:
            name (str | Callable | None): 名字；``@registry.register`` 这种不写参数的用法里，
                Python 会把函数本身传进来。
            fn (Callable | None): 要登记的函数（``registry.register("名字", 函数)`` 的写法）。

        Returns:
            Callable: 原样返回被登记的函数，所以能当装饰器用。

        Raises:
            ValueError: 名字为空，或这个名字已经注册过。
        """
        if callable(name) and fn is None:  # @registry.register
            return self._do_register(name.__name__, name)
        if fn is not None:  # registry.register("名字", 函数)
            return self._do_register(name or fn.__name__, fn)

        def decorator(func):  # @registry.register("名字")
            return self._do_register(name or func.__name__, func)

        return decorator

    def _do_register(self, name, fn):
        if not isinstance(name, str) or not name:
            raise ValueError(f"{self.kind}的名字必须是非空字符串，收到 {name!r}")
        if name in self._items:
            raise ValueError(f"{self.kind}名字 {name!r} 已经注册过了；换个名字再注册")
        self._items[name] = fn
        return fn

    def get(self, name: str):
        """按名字取函数。

        Raises:
            KeyError: 名字没注册过；报错里会列出已经注册的名字。
        """
        if name not in self._items:
            raise KeyError(f"没有注册名为 {name!r} 的{self.kind}；已注册：{self.names()}")
        return self._items[name]

    def resolve(self, spec: str | Callable) -> tuple[str, Callable]:
        """把"名字或函数"统一成 ``(名字, 函数)``，让调用方两种都收。

        Args:
            spec (str | Callable): 注册过的名字，或者任何 callable。

        Returns:
            tuple: ``(名字, 函数)``。

        Raises:
            KeyError: 名字没注册过。
            TypeError: 既不是字符串也不是可调用对象。

        Examples:
            >>> from llm_kernel.nn.registry import Registry
            >>> METRICS = Registry("指标")
            >>> METRICS.register("accuracy", lambda y_hat, y: 1.0)
            <function <lambda> at ...>
            >>> METRICS.resolve("accuracy")[0]
            'accuracy'
            >>> def my_metric(y_hat, y):
            ...     return 0.0
            >>> METRICS.resolve(my_metric)[0]
            'my_metric'
        """
        if isinstance(spec, str):
            return spec, self.get(spec)
        if not callable(spec):
            raise TypeError(f"{self.kind}只能是名字或可调用对象，收到 {type(spec).__name__}")
        return getattr(spec, "__name__", repr(spec)), spec

    def names(self) -> list:
        """已注册的名字，按注册顺序。"""
        return list(self._items)

    def __contains__(self, name) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"<Registry {self.kind}: {self.names()}>"
