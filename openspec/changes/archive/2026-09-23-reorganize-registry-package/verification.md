# Verification

## 结果

本 change 完成 Registry 内部模块迁移，公共 `ai_qec.notebook_api` 导出集合、零注册导入行为和可选 runtime 延迟加载行为保持不变。相关测试、Registry 子系统静态检查、干净 wheel 安装检查及 OpenSpec 严格验证均通过。

## 代码与导入验证

- `find ai_qec/registry -maxdepth 1 -type f`：仅包含 `bootstrap.py`、`catalog.py`、`core.py`、`validation.py`。
- `find ai_qec/registry -name '__init__.py'`：无输出。
- `rg 'ai_qec\.(config_validation|implementations|registries)|from ai_qec\.registry import' ai_qec tests README.md`：无旧路径引用。
- `python -m compileall -q ai_qec tests`：通过。
- `git diff --check`：通过。

## 测试

- `python -m unittest tests.unit.test_registry tests.unit.test_packaging`：通过，运行 $16$ 项，跳过 $1$ 项可选 runtime 测试。
- 排除与本 change 无关的论文 Notebook 文件后运行其余单元测试：通过，运行 $97$ 项，跳过 $29$ 项可选依赖测试。
- `python -m unittest discover -s tests/integration`：运行 $11$ 项，全部因本机缺少可选 runtime 而跳过。
- 直接导入 `ai_qec.notebook_api`：公共导出数仍为 $130$，Registry 项目数为 $0$，且 `torch`、`stim`、`pymatching`、`qiskit`、`cudaq` 均未加载。

全量 `python -m unittest discover -s tests/unit` 运行 $104$ 项时有 $1$ 项既有失败：`test_notebook_is_valid_v4_json` 检测到 `paper/srcs/ai_for_qec_workflow.ipynb` 已保存输出。该 Notebook 在本 change 前后均无工作区 diff，本 change 未修改它。

## 静态检查

- `ruff check --no-cache ai_qec/registry`：通过，$0$ 项错误。
- `pyright ai_qec/registry`：通过，$0$ 项错误、$0$ 项警告。

全仓审计发现既有基线未清零：Ruff 报告 $61$ 项旧代码格式/规则问题，Pyright 在缺少 NumPy、PyTorch 等可选依赖的当前环境报告 $2953$ 项既有类型问题。这些结果不由模块迁移引入，未在本 change 中扩大范围修复。

## Wheel 验证

从仅包含当前 `ai_qec` 源码、`README.md` 与 `pyproject.toml` 的 `/tmp` 干净副本运行：

```bash
python -m pip wheel <clean-source> --no-deps --no-build-isolation --wheel-dir <wheel-dir>
python -m pip install --no-deps --target <install-dir> <wheel>
```

结果：

- wheel SHA-256：`94e699caa010a343c5a2bdb6021e456fc553f05bbbd2a623bd012d20e1ba17d0`。
- wheel 包含四个 `ai_qec/registry/*.py` 叶子模块和 `ai_qec/notebook_api.py`。
- wheel 不包含 `ai_qec/config_validation.py`、`ai_qec/implementations.py`、`ai_qec/registries.py` 或旧的 `ai_qec/registry.py`。
- 在仓库外安装目录导入成功：公共导出 $130$ 个、四个叶子模块均可导入、旧顶层模块数为 $0$、导入后注册数为 $0$、可选 runtime 加载数为 $0$。

工作区直接构建曾观察到历史 `build/` 目录把已删除模块带入 wheel；干净源码构建未复现，说明这是未清理的本地构建缓存而不是当前源码或 setuptools namespace 配置问题。本 change 未删除用户已有构建产物。

## OpenSpec

- `openspec validate reorganize-registry-package --strict --no-interactive`：通过。
- `openspec validate --all --strict --no-interactive`：$11$ 项通过，$0$ 项失败。
