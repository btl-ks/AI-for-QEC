# Spec Delta

## Purpose

定义 AI 与经典解码器之间公平、可审计且具有统计意义的科学比较协议，并用显式 Accuracy Gate 控制何时可以冻结模型和进入性能研究。

## ADDED Requirements

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
