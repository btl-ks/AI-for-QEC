# Spec Delta

## ADDED Requirements

### Requirement: 配对相对非劣效 Accuracy Gate
`comparison_rule` 为 `paired-relative-non-inferiority` 时，`tolerance` SHALL 表示相对容差 δ，且 MUST 为非负有限数，否则配置 MUST 在创建任何 Experiment 或数据集之前失败。Gate SHALL 使用与 `paired-non-inferiority` 相同的配对失败列联表，估计 `LER_AI − (1+δ)·LER_baseline` 的置信区间；δ = 0 时该区间 SHALL 与 `paired-non-inferiority` 的配对差异区间相同。只有当区间上界不超过 0 时 Gate SHALL 判定 PASS。判定、列联表、δ、区间与 `LER_AI / LER_baseline` 的点估计 SHALL 持久化为可复核证据，rationale SHALL 包含比值点估计、区间与 δ。

#### Scenario: 低错误率下相对差距超过容差
- **WHEN** 在同一组 10 000 shots 上 baseline LER 为 0.0384、AI LER 为 0.0431（比值约 1.12），δ = 0.15，置信水平 95%，且区间上界大于 0
- **THEN** Gate SHALL 返回 FAIL，即使同一数据在绝对容差 0.02 下为 PASS

#### Scenario: 高错误率下相对差距在容差内
- **WHEN** baseline LER 为 0.5435、AI LER 为 0.5760（比值约 1.06），δ = 0.15，且 `LER_AI − 1.15·LER_baseline` 的区间上界不超过 0
- **THEN** Gate SHALL 返回 PASS，rationale SHALL 给出比值点估计、区间与 δ

#### Scenario: 边界上的判定频率符合名义水平
- **WHEN** 在真实 `LER_AI = (1+δ)·LER_baseline` 的配对分布下以 10 000 shots 重复抽样，置信水平为 95%
- **THEN** PASS 的比例 SHALL 落在 1.5% 到 3.5% 之间（单侧名义水平为 2.5%）

#### Scenario: 相对容差为负
- **WHEN** 配置声明 `paired-relative-non-inferiority` 且 `tolerance = -0.1`
- **THEN** 预检 SHALL 报告 `accuracy_gate.tolerance` 并失败，且 MUST NOT 创建 `runs/` 或 `datasets/` 中的任何条目
