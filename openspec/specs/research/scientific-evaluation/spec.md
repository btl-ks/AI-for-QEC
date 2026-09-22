# research/scientific-evaluation Specification

## Purpose
定义 AI 与经典解码器之间公平、可审计且具有统计意义的科学比较协议，并用显式 Accuracy Gate 控制何时可以冻结模型和进入性能研究。

## Requirements

### Requirement: 科学比较使用相同协议
AI 与 baseline decoder SHALL 使用相同 QEC、Noise、测试 DatasetArtifact、shot 集合、observable truth、停止语义和 LER 定义。

#### Scenario: 比较 AI 与 MWPM
- **WHEN** 科学评估同时运行 AI decoder 和 MWPM baseline
- **THEN** 两者 SHALL 引用相同 DatasetArtifact 和 sample identities

### Requirement: LER 是主要指标
ScientificEvaluationResult SHALL 报告 logical error count、shots、Logical Error Rate 和置信区间，并 SHALL 显式记录 timeout、未收敛或无效样本的处理方式。

#### Scenario: 产生评估结果
- **WHEN** decoder 完成一个有效测试集的科学评估
- **THEN** 结果 MUST 包含 LER 的分子、分母、点估计和置信区间信息

### Requirement: Accuracy Gate 规则预先声明
AccuracyGateSpec SHALL 在评估前声明 baseline、primary metric、统计规则和允许 tolerance；ScientificAcceptanceResult SHALL 持久化 `PASS` 或 `FAIL`、判定依据与输入 Artifact 引用。

#### Scenario: Gate 判定
- **WHEN** AI 与 baseline 的完整评估结果可用
- **THEN** gate SHALL 只依据预先登记的规则作出 PASS 或 FAIL，并保存可复核证据

### Requirement: 性能优化受双重 Gate 约束
未经 Gate A 通过的模型 MUST NOT 被标记为正式性能研究基线；量化、剪枝、精度变换或硬件映射后的候选 SHALL 通过 Gate B 才能成为可接受部署结果。

#### Scenario: 优化候选更快但准确率退化
- **WHEN** 优化候选达到性能目标但未通过 Gate B
- **THEN** 系统 SHALL 保留性能结果，同时 MUST NOT 将该候选标记为可接受部署结果

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
