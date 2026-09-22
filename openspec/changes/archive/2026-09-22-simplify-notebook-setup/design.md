# Design

## Context

动机见 proposal.md。首个 cell 的 `sys.path` 修改是因为内核环境里没有安装 `ai_qec`。本机 `quantum` 环境现已通过 `pip install --no-deps -e .` 安装本项目，因此导入不再需要路径 glue；剩下需要的只有 `LocalNotebookPlatform(PROJECT_ROOT)` 用到的项目根目录。

## Goals / Non-Goals

**Goals:**

- setup cell 只保留导入与根目录定位两句。
- 根目录定位可测试、失败明确，并且与 Notebook 的工作目录无关。

**Non-Goals:**

- 不让 `LocalNotebookPlatform` 隐式推断根目录；Notebook 仍显式传入 `PROJECT_ROOT`。

## Decisions

### 1. 以 `openspec/config.yaml` 作为项目根目录标记

它只存在于本仓库根目录，比 `pyproject.toml` 更不容易与父目录中的其他项目混淆。先从 `start`（默认 `Path.cwd()`）向上查找，因此 VS Code/Jupyter 默认把 cwd 设为 Notebook 所在目录时可以直接找到。

### 2. 退回 editable 安装位置，否则失败

若 cwd 不在仓库内（例如从其他目录启动内核），检查 `ai_qec` 包的上一级目录是否含该标记；这只在 editable 安装或直接运行源码时成立。wheel 安装到 site-packages 时没有该标记，函数抛出 `FileNotFoundError`，而不是返回一个会让 runtime 把 `datasets/`、`runs/` 写到错误位置的目录。

替代方案：在 Notebook 中写 `Path(qec.__file__).parents[1]`。否决原因：依赖安装方式，且在 wheel 安装时会静默返回 site-packages。

### 3. 只做标准库实现

`find_project_root` 位于 `ai_qec/utils/paths.py`，只用 `pathlib`，保持导入 facade 不加载任何框架。

## Risks / Trade-offs

- [内核环境未安装本项目时导入失败] → README 与 paper/README 说明 editable 安装；单元测试在仓库根目录运行，仍可导入。
