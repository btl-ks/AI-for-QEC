# Proposal

## Why

论文 Notebook 用 `copy.deepcopy` 加逐层下标赋值的 `paper_config()` 从 `CONFIG` 派生 22 个网格点，另有 `PROFILES`、`HYPERPARAMETERS_BY_DISTANCE` 与 `GRID` 辅助结构。展开网格的机制与论文无关，却写在 Notebook 里；逐层下标还让类型检查器把 `dict[str, object]` 的嵌套访问标为错误，且拼错字段名时会静默新增一个无人读取的键。

## What Changes

- 新增通用配置网格展开 API（`qec.config_grid`、`qec.with_overrides`、`qec.GridPoint`）：按轴的笛卡尔积从基础配置生成独立配置，支持所有点共享的覆盖和按某个轴取值耦合的覆盖，并用名称模板设置 `experiment.name`。
- 覆盖只能指向基础配置中已存在的字段；重复覆盖、耦合覆盖缺少轴取值、名称重复都会在返回任何配置前失败；基础配置不被修改。
- Notebook 删除 `paper_config`、`PROFILES`、`HYPERPARAMETERS_BY_DISTANCE`，改为声明两个网格（`paper`、`smoke`）；论文参数仍全部写在 Notebook 中。
- **BREAKING（运行产物）**：实验名由 `…-l4-p008` 改为 `…-l4-p0.08`，Experiment 身份随之变化，首次 Run All 会重新训练 22 个点；数据集的 DatasetKey 不含实验名，已提交的数据集继续复用。旧 `runs/` 记录保留不删除。

非目标：不把 Torlai–Melko 的具体参数放进库；不改变 runtime、训练或评估行为。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `governance/configuration-registry`: 增加配置网格展开只能覆盖已声明字段、冲突与名称重复必须在展开前失败的要求。

## Impact

- 新增 `ai_qec/experiment/grid.py`（纯标准库），从 `ai_qec.notebook_api` 导出。
- 修改 `paper/srcs/ai_for_qec_workflow.ipynb` 的 config、execute cells 与说明；修改 `tests/unit/test_paper_notebook.py`，新增 `tests/unit/test_config_grid.py`。
