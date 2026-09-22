# Spec Delta

## ADDED Requirements

### Requirement: Notebook 定义 cells 无副作用
Notebook 中定义配置、编排函数和辅助函数的 cells SHALL 在导入或执行时不创建 runtime、Run、dataset、model、metric 或 performance Artifact；启动实验的 cells SHALL 用 cell 标签 `run-experiment` 显式标记，并 SHALL 只通过显式构造的 runtime 调用 `run_paper`。

#### Scenario: 只执行定义 cells
- **WHEN** 从干净内核执行全部未标记 `run-experiment` 的 code cells
- **THEN** 执行 SHALL 成功完成，SHALL 定义 `CONFIG` 与 `run_paper`，且 SHALL NOT 创建 runtime 或任何 Artifact

#### Scenario: Run All
- **WHEN** 用户在安装运行时依赖的内核中执行全部 cells
- **THEN** 标记 `run-experiment` 的 cells SHALL 构造本地 runtime 并按声明网格执行 `run_paper`

### Requirement: Torlai–Melko 2017 复现协议
论文 Notebook SHALL 复现 Torlai & Melko (PRL 119, 030501, 2017) 的 toric code 相位翻转实验：对声明网格中的每个 (L, p) 点独立训练联合 RBM，以算法 1 解码 test split，并在同一 test split 上与 MWPM 比较；Notebook SHALL 展示 P_fail–p 曲线（图 3）与 RBM 残余同调类直方图（图 4），并 SHALL 标明所用超参数、样本数和与论文设置的差异。

#### Scenario: 执行论文网格
- **WHEN** Notebook 以 `paper` profile 执行
- **THEN** 每个 (L, p) 点 SHALL 拥有独立 Experiment、ScientificEvaluationResult 与 ScientificAcceptanceResult，且图 3 SHALL 同时显示两个 decoder 的点估计与置信区间

#### Scenario: 冒烟 profile
- **WHEN** Notebook 以 `smoke` profile 执行
- **THEN** Notebook SHALL 使用小规模配置走完同一编排路径，并 SHALL 明确说明其结果不能作为科学结论

## REMOVED Requirements

### Requirement: contracts-only Notebook 可安全执行
**Reason**: 仓库已提供满足 `NotebookPlatform` 的本地 runtime，Run All 会按设计执行实验并产生 Artifact，“Run All 不产生任何 Artifact”不再成立。
**Migration**: 由“Notebook 定义 cells 无副作用”取代：定义 cells 仍保证无副作用，实验 cells 以 `run-experiment` 标签显式标记。
