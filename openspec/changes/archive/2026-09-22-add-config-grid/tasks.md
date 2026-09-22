# Tasks

## 1. 实现

- [x] 1.1 新增 `ai_qec/experiment/grid.py`（`with_overrides`、`config_grid`、`GridPoint`）并从 `ai_qec.notebook_api` 导出，以 `tests/unit/test_config_grid.py` 验证笛卡尔积顺序、耦合覆盖、不修改基础配置、拼错路径、覆盖冲突、耦合缺值与名称重复均失败
- [x] 1.2 用两个 `qec.config_grid` 声明替换 Notebook 中的 `paper_config`、`PROFILES`、`HYPERPARAMETERS_BY_DISTANCE`，以 `tests/unit/test_paper_notebook.py` 验证论文网格 22 个点、超参数按 L 耦合、smoke 网格与编排顺序

## 2. 验证与归档

- [x] 2.1 在 `quantum` 与基础环境运行全部测试，并在 `quantum` 内核中执行 Notebook（paper profile），确认无错误、22 个新 Experiment 完成且数据集全部复用，把结果写入 `verification.md`
- [x] 2.2 运行 `openspec validate --all --strict --no-interactive` 后归档
