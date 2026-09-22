# Verification

## Scope

验证 Python 分发从仓库根目录 Hatchling package 迁移到 setuptools `src` layout 和 PEP 420 隐式 namespace package，同时保持正式公共入口 `ai_qec.notebook_api` 的 130 个导出及领域行为不变。此次验证不改变科研语义、runtime backend、Registry 内容或 capability 科学状态。

## Environment

| 项目 | 值 |
| --- | --- |
| 构建与隔离安装 Python | 3.13.12 |
| 完整 runtime 测试 Python | 3.12.14 (`quantum`) |
| setuptools（构建环境） | 80.10.2 |
| NumPy | 2.5.3 |
| PyTorch | 2.14.0 |
| Stim | 1.16.0 |
| PyMatching | 2.4.0 |
| Matplotlib | 3.11.2 |
| OpenSpec | 1.13.1 |

主机没有可用 CUDA，因此 CUDA-only 单元测试 7 项、集成测试 2 项按既有条件跳过；CPU runtime 路径均实际执行。

## Layout and build evidence

命令：

```bash
test ! -d ai_qec
find src/ai_qec tests -type f -name '__init__.py' -print
python -m compileall -q src/ai_qec tests
uv lock --offline
uv lock --check
uv build --wheel --no-build-isolation
```

结果：仓库根目录不存在第二份 `ai_qec` 源码树；`src/ai_qec` 与 `tests` 中 `__init__.py` 数量为 0；64 个源码模块编译成功；锁文件与项目元数据同步；setuptools 成功构建 `dist/ai_for_qec-0.1.0.dev0-py3-none-any.whl`。

最终 wheel checksum：

```text
sha256:56a57c8e0fac4ed1cb3daf316c6eb6275af58e1e6760475e0f3e7f7521fd797f
```

机器可读检查结果保存在 `evidence/wheel-inspection.json`。

## Editable-install tests

先从 `quantum` Python 建立继承 runtime dependencies 的临时虚拟环境，再执行真实 editable 安装；测试命令没有设置 `PYTHONPATH`：

```bash
/home/zephy/miniconda3/envs/quantum/bin/python -m venv --system-site-packages <tmp>/venv
<tmp>/venv/bin/python -m pip install --no-deps --no-build-isolation -e .
env -u PYTHONPATH <tmp>/venv/bin/python -m unittest discover -s tests/unit
env -u PYTHONPATH <tmp>/venv/bin/python -m unittest discover -s tests/integration
```

结果：

- unit：102 项运行，全部通过，7 项仅因无 CUDA 跳过；
- integration：11 项运行，全部通过，2 项仅因无 CUDA 跳过；
- `ai_qec.__file__ is None`、`ai_qec.__spec__.origin is None`，且 namespace path 唯一指向 `src/ai_qec`；
- 首轮回归发现 `find_project_root` 仍依赖旧目录深度，随后改为向上查找 `openspec/config.yaml`，最终完整回归通过。

packaging contract 定向验证：

```bash
env -u PYTHONPATH <tmp>/venv/bin/python \
  -m unittest discover -s tests/unit -p 'test_packaging.py' -v
```

结果：5 项全部通过，覆盖唯一 src 树、零初始化文件、setuptools namespace 配置、130 个公共导出快照及深层叶子模块导入。

## Isolated wheel-install evidence

在仓库外新建不继承系统依赖的 Python 3.13 虚拟环境，仅安装所构建 wheel：

```bash
python -m venv <tmp>/venv
<tmp>/venv/bin/python -m pip install --no-deps \
  dist/ai_for_qec-0.1.0.dev0-py3-none-any.whl
cd <tmp>
env -u PYTHONPATH <tmp>/venv/bin/python -c '<import and assertion script>'
```

实际输出：

```text
distribution 0.1.0.dev0
namespace_origin None
facade <tmp>/venv/lib/python3.13/site-packages/ai_qec/notebook_api.py
public_exports 130 f2516ff66e76ca2ca8efb36ac367a18a797e387492081bea63f0008578a1d9b5
optional_runtime_loaded []
python_modules 64
init_files []
```

同时成功导入 `ai_qec.data.datasets.artifact`、`ai_qec.evaluation.scientific.result` 与 `ai_qec.training.executors.protocol`。这证明 wheel 不依赖仓库根目录、深层模块由 namespace discovery 收录，且导入 facade 不会提前加载 torch、stim、pymatching、qiskit 或 cudaq。

## Documentation and governance

以下检查通过：

```bash
rg '(^|[` ])ai_qec/' README.md openspec/capabilities.yaml pyproject.toml
python -c '<validate openspec/capabilities.yaml against capabilities.schema.json>'
openspec validate adopt-src-namespace-packaging --strict --no-interactive
openspec validate --all --strict --no-interactive
```

结果：当前 README、构建配置和 capability evidence 不再引用旧物理源码根；10 个 capability 条目通过 JSON Schema；change 严格验证通过；归档前全部 10 个 OpenSpec items 通过严格验证。
