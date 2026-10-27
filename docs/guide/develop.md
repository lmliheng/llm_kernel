# 开发与发布

## 本地环境

```bash
git clone https://github.com/lmliheng/llm_kernel.git
cd llm_kernel
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"       # 开发 + 文档依赖
pip install "llm_kernel[plot]"     # 想跑画图测试再加上 matplotlib
# 想跑 nn 的测试还得有 torch（可以用 CPU 版）：
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**别在 `site-packages` 里改代码**。用可编辑安装（`pip install -e .`）指向仓库，
改完立刻生效，`git status` 也不会被构建垃圾污染。

## 跑测试

```bash
pytest                      # 全跑；没有 torch / matplotlib 的相关用例会自动跳过
pytest tests/test_math.py -v
pytest -k normal
```

测试分两类，缺一不可：

- `tests/test_*.py`：行为测试（边界值、异常消息、旧名字兼容、惰性导入）。
- `tests/test_docstrings.py`：把每个公开模块 docstring 里的 `Examples:` 当测试跑，
  保证文档不会和代码一起腐烂。

CI 见 `.github/workflows/tests.yml`：Python 3.9 / 3.12 / 3.13 三个版本跑基础测试，
另有一个任务装 torch 跑 `nn` 的完整用例。

## 构建校验（别跳过）

```bash
python -m build                                   # 生成 dist/*.whl 和 *.tar.gz
twine check dist/*                                # 检查元数据
python -m zipfile -l dist/*.whl                   # ★ 确认 wheel 里真的有 llm_kernel/…
python -m venv /tmp/v && /tmp/v/bin/pip install dist/*.whl
/tmp/v/bin/python -c "import llm_kernel; print(llm_kernel.__version__, llm_kernel.math.normal(0.0))"
```

第三步是重点：**`twine check` 通过不代表包里有代码**。曾经出现过 wheel 里只有一个
`dist-info` 目录、`twine check` 照样 PASSED、`pip install` 也提示安装成功、
装完 `import llm_kernel` 却 `ModuleNotFoundError` 的情况（根因是 `pyproject.toml` 的
`packages.find` 和实际目录结构不匹配）。所以发布前一定要看到 `llm_kernel/` 下的真实文件，
并且真的 import 一次。

## 本地预览文档

```bash
mkdocs serve            # http://127.0.0.1:8000
mkdocs build --strict   # CI 用的命令，任何警告都算失败
```

API 页面由 docstring 自动生成（mkdocstrings），所以**写 docstring 就是写文档**：
函数写 Google 风格的 `Args` / `Returns` / `Raises` / `Examples`，
示例会被 `tests/test_docstrings.py` 真的执行。

## 加代码的固定流程

1. 改/加代码（配方见 [API 设计说明](design.md#extension-recipes)）；
2. 补测试；
3. 跑 `pytest` 和 `mkdocs build --strict`；
4. 在 `docs/changelog.md` 记录改动；
5. 提交。

## 发布

版本号**只有一处**：`llm_kernel/__init__.py` 里的 `__version__`。
`pyproject.toml` 用 `dynamic = ["version"]` 读它，所以不存在两处不一致的问题。

```bash
# 1. 改版本号（例如 0.2.0 → 0.2.1）并更新 CHANGELOG
# 2. 构建并自检
python -m build && twine check dist/* && python -m zipfile -l dist/*.whl | grep llm_kernel/
# 3. 提交、打 tag、创建 Release
git commit -am "chore: release 0.2.1"
git tag v0.2.1
git push origin main --tags
# 4. 在 GitHub 上创建 Release（或手动触发 Actions 里的 Upload Python Package）
```

`publish.yml` 会做的事：构建 sdist/wheel → `twine check` → 装进干净环境 import 一次 →
校验 **tag 与 `__version__` 一致** → 上传 PyPI。

### PyPI 认证两种方式

| 方式 | 怎么配 | 注意 |
| --- | --- | --- |
| Trusted Publishing（推荐） | 在 PyPI 的 [Publishing](https://pypi.org/manage/account/publishing/) 里登记：仓库 `lmliheng/llm_kernel`、workflow `publish.yml`、environment `pypi` | 不用保存任何密钥；工作流走 OIDC |
| API Token | 仓库 Settings → Secrets 加 `PYPI_API_TOKEN` | 工作流里已经接好，配了就优先用它 |

### 发布前必须知道的两件事

1. **同一个版本号不能重发**。PyPI 会直接拒绝 `File already exists`；
   已经发出去的版本只能 *yank*（隐藏），删不掉。
2. **GitHub Pages 的部署只允许从默认分支（main）触发**，
   所以 `docs.yml` 要在 main 上才会有效果；另外首次需要在
   Settings → Pages → Source 里选 **GitHub Actions**。

## 文档站

```bash
mkdocs serve          # 本地
```

推送到 main 后 `.github/workflows/docs.yml` 会自动构建并部署到
<https://lmliheng.github.io/llm_kernel/>。
