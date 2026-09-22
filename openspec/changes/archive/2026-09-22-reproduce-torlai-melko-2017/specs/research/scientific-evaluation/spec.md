# Spec Delta

## ADDED Requirements

### Requirement: LER 区间与无效样本策略
每个 decoder 的 LER SHALL 以逻辑失败数除以 test shots 计算，并 SHALL 使用 Wilson score 区间报告声明置信水平的置信区间；`invalid_sample_policy` 为 `count-as-failure` 时，超时或未收敛样本 SHALL 计为逻辑失败并单独计数，为 `fail` 时出现任何无效样本 MUST 使评估失败。

#### Scenario: AI decoder 有超时样本
- **WHEN** 策略为 `count-as-failure` 且 RBM decoder 有 m 个超时样本
- **THEN** 该 decoder 的失败数 SHALL 包含这 m 个样本，且 `timeout_count` SHALL 等于 m

### Requirement: 报告残余逻辑类分布
DecoderEvaluation SHALL 报告有效预测样本上“物理错误与恢复之积”的残余逻辑类计数，类别由逻辑可观测量翻转位串表示，其中全零类为成功恢复。

#### Scenario: 复现同调类直方图
- **WHEN** 评估 toric code 上的 decoder
- **THEN** 结果 SHALL 包含 `00`、`10`、`01`、`11` 四个类别的计数，其和 SHALL 等于有效预测样本数

### Requirement: 配对非劣效 Accuracy Gate
`comparison_rule` 为 `paired-non-inferiority` 时，Gate SHALL 在同一 shot 集合上计算 AI 与 baseline 的配对失败列联表，用 Newcombe 配对差异区间估计 `LER_AI − LER_baseline`，并且只有当区间上界不超过预先声明的 `tolerance` 时判定 PASS；判定、列联表、区间和 tolerance SHALL 持久化为可复核证据。未知规则或指标 MUST 使 Gate 失败而不是给出默认判定。

#### Scenario: AI 显著劣于 baseline
- **WHEN** 配对差异区间下界大于 `tolerance`
- **THEN** Gate SHALL 返回 FAIL，rationale SHALL 包含点估计、区间和 tolerance

#### Scenario: Gate 规则未知
- **WHEN** AccuracyGateSpec 声明不受支持的 `comparison_rule`
- **THEN** Gate SHALL 抛出明确错误且 MUST NOT 产生 ScientificAcceptanceResult
