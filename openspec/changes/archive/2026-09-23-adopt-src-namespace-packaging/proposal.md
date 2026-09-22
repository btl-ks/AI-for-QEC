# Proposal

## Why

当前 Python 源码位于仓库根目录并依赖多层 `__init__.py` 聚合导出，源码树与安装后的包边界不够清晰。项目需要采用标准 `src` 布局与 setuptools 隐式 namespace package 发现，使源码只能通过明确安装后被消费，并消除仅为包发现而维护的初始化文件。

## What Changes

- 将可安装源码从 `ai_qec/` 迁移到 `src/ai_qec/`。
- 将构建后端从 Hatchling 切换为 setuptools，并启用 `src` 根目录下的隐式 namespace package 自动发现。
- 移除源码包与测试目录中的全部 `__init__.py`，把公共 facade 的聚合导入改为直接引用叶子模块。
- 保持已声明公共入口 `import ai_qec.notebook_api as qec` 及其导出集合可用。
- 更新测试、文档、能力证据路径和构建锁文件，并验证 wheel 安装后的导入行为。
- **BREAKING**：未由 `ai_qec.notebook_api` 声明的包级便捷导入（例如 `from ai_qec.data import QECBatch`）不再提供；调用方必须使用公共 facade 或具体叶子模块。
- 不改变数据、训练、解码、评估、生命周期或 Registry 的科学语义；不新增 GPU、分布式、FPGA、ASIC 或 QPU 能力。

## Capabilities

### New Capabilities

- `governance/python-packaging`: 规定 `src` 布局、setuptools namespace package 发现、零 `__init__.py` 约束和已安装公共 API 的兼容性。

### Modified Capabilities

无。

## Impact

影响 `pyproject.toml`、`uv.lock`、Python 源码物理路径、内部聚合导入、测试发现命令、README、能力证据路径及打包验证。公开 Notebook 导入路径保持不变；依赖仓库根目录隐式进入 `sys.path` 或依赖内部包级聚合导出的用法需要迁移。
