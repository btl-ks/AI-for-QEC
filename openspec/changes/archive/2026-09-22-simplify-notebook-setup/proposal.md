# Proposal

## Why

`ai_for_qec_workflow.ipynb` 的首个 code cell 用十几行代码向上搜索项目根目录、手动修改 `sys.path` 后才能导入 `ai_qec`。这些 glue 只是因为包没有安装在内核环境中，并且 `PROJECT_ROOT` 的定位逻辑写在 Notebook 里。

## What Changes

- 在 `ai_qec.notebook_api` 中新增 `find_project_root(start=None)`：从起点（默认当前目录）向上查找含 `openspec/config.yaml` 的目录；找不到时退回到 editable 安装的源码目录；两者都失败则抛出 `FileNotFoundError`，不猜测路径。
- Notebook 首个 code cell 缩减为导入 facade 与 `PROJECT_ROOT = qec.find_project_root()` 两句；`copy`、`Mapping` 移到使用它们的 cell；删除 Python 3.11+ 不需要的 `from __future__ import annotations`。
- 文档说明内核环境需以 editable 方式安装本项目（`pip install -e ".[runtime]"`）。

非目标：不改变 runtime、编排顺序、配置或任何实验结果。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

无。spec 级行为不变（Notebook 仍只通过公共 facade 调用项目代码），本 change 设置 `skip_specs: true`。

## Impact

- 新增 `ai_qec/utils/paths.py`，并从 `ai_qec.notebook_api` 导出 `find_project_root`（纯标准库，不加载任何框架）。
- 修改 `paper/srcs/ai_for_qec_workflow.ipynb` 的 setup、config、orchestration cells 与 `paper/README.md`。
- 新增 `tests/unit/test_paths.py`。
