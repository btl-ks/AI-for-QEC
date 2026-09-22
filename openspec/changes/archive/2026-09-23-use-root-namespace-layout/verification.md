# Verification

## Scope

验证源码从 `src/ai_qec` 恢复到仓库根目录 `ai_qec`，同时继续使用 setuptools、PEP 420 隐式 namespace package 和零 `__init__.py`。验证覆盖根目录直接导入、受限 package discovery、完整 CPU runtime 回归和仓库外 wheel 安装；不改变领域 contract、Registry 或科学能力。

## Layout and direct-import evidence

命令：

```bash
test -d ai_qec
test ! -d src/ai_qec
test ! -d src
find ai_qec tests -type f -name '__init__.py' -print
python -m compileall -q ai_qec tests
python -c 'import ai_qec, ai_qec.notebook_api as qec; print(ai_qec.__spec__.origin, qec.__file__, len(qec.__all__))'
```

结果：唯一源码树位于根目录 `ai_qec`，生成的 `src` 目录已清理，整个项目 `__init__.py` 数量为 0。未安装状态从仓库根目录导入成功：namespace origin 为 `None`，facade 解析到 `/home/zephy/workspace/QEC/ai_qec/notebook_api.py`，公共导出数量为 130。

## Packaging contract and runtime tests

命令：

```bash
python -m unittest discover -s tests/unit -p 'test_packaging.py' -v
env -u PYTHONPATH MPLCONFIGDIR=/tmp/qec-matplotlib \
  /home/zephy/miniconda3/envs/quantum/bin/python -m unittest discover -s tests/unit -v
env -u PYTHONPATH MPLCONFIGDIR=/tmp/qec-matplotlib \
  /home/zephy/miniconda3/envs/quantum/bin/python -m unittest discover -s tests/integration -v
```

结果：

- packaging contract：6 项全部通过，覆盖唯一根目录源码树、受限 `ai_qec*` discovery、零初始化文件、根目录直导入、叶子模块导入和公共导出快照；
- unit：103 项运行，全部通过，7 项仅因主机无 CUDA 跳过；
- integration：11 项运行，全部通过，2 项仅因主机无 CUDA 跳过。

## Build and isolated wheel evidence

命令：

```bash
uv lock --offline
uv lock --check
uv build --wheel --no-build-isolation
python -c '<inspect wheel members and checksum>'
python -m venv <tmp>/venv
<tmp>/venv/bin/python -m pip install --no-deps --force-reinstall \
  dist/ai_for_qec-0.1.0.dev0-py3-none-any.whl
cd <tmp>
env -u PYTHONPATH <tmp>/venv/bin/python -c '<import and assertion script>'
```

wheel 检查结果：

```text
wheel_sha256 1b13c0e129946054679c178e0dc02aacc0391c40e2825a1d77014a09e9483e43
python_modules 64
init_files []
unexpected_python_packages []
```

隔离安装实际输出：

```text
distribution 0.1.0.dev0
namespace_origin None
facade <tmp>/venv/lib/python3.13/site-packages/ai_qec/notebook_api.py
public_exports 130 f2516ff66e76ca2ca8efb36ac367a18a797e387492081bea63f0008578a1d9b5
optional_runtime_loaded []
```

wheel 中所有 64 个 Python 模块均位于 `ai_qec/`，未收录 `tests`、`docs`、`openspec` 或其他 Python package。机器可读结果保存在 `evidence/wheel-inspection.json`。

## Documentation and governance

命令：

```bash
rg 'src/ai_qec' README.md openspec/capabilities.yaml docs/AI_for_QEC_Codex_Architecture_Context.md
python -c '<validate openspec/capabilities.yaml against capabilities.schema.json>'
git diff --check
openspec validate use-root-namespace-layout --strict --no-interactive
openspec validate --all --strict --no-interactive
```

结果：当前 README、capability evidence 和架构背景不再引用 `src/ai_qec`；10 个 capability 条目通过 JSON Schema；change 与全部 11 个归档前 OpenSpec items 通过严格验证。
