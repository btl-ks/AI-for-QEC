# Spec Delta

## ADDED Requirements

### Requirement: 跨 Experiment 复用已完成 Stage
训练、科学评估与性能评估 Stage 开始时，若当前 Attempt 的恢复源不能提供可复用输出，系统 SHALL 在所有 Experiment 中查找同名、状态为 `completed`、全部输出 checksum 校验通过、且 Stage 复用键相同的 Stage，找到时 SHALL 复用其输出而不重新计算。Stage 复用键 SHALL 由 Stage 名、该 Stage 的输入 Artifact 引用，以及完整规范化配置去掉下列字段后的内容共同决定：

- 训练：`experiment.name`、`model.decoding`、`scientific_evaluation`、`accuracy_gate`、`performance`
- 科学评估：`experiment.name`、`accuracy_gate`、`performance`
- 性能评估：`experiment.name`、`accuracy_gate`

未列出的字段，包括今后新增的字段，SHALL 进入复用键。Accuracy Gate 与可视化 Stage MUST NOT 跨 Experiment 复用。复用时，新 Stage 的记录 SHALL 包含来源 Experiment、来源 Attempt 与复用键；复用的 Artifact SHALL 以其原引用（位置与 checksum）作为 Stage 输出，MUST NOT 被复制或重新登记为当前 Attempt 的产物。Experiment 身份仍 SHALL 由完整规范化配置的摘要决定。

#### Scenario: 只修改 Accuracy Gate
- **WHEN** 配置 A 已完成全部 Stage，配置 B 与 A 只在 `accuracy_gate` 上不同，用户执行配置 B
- **THEN** 系统 SHALL 为 B 创建新的 Experiment；B 的训练与科学评估 Stage SHALL 复用 A 的输出且不训练任何 epoch，Accuracy Gate SHALL 按 B 的规则重新计算；B 的 Gate 为 PASS 且 A 有已完成的性能评估时，性能评估 SHALL 复用 A 的输出

#### Scenario: 修改训练字段
- **WHEN** 配置 B 与已完成的配置 A 只在 `training.epochs` 上不同
- **THEN** B 的训练 Stage MUST NOT 复用 A 的输出，SHALL 训练新模型

#### Scenario: 只修改实验名
- **WHEN** 配置 B 与已完成的配置 A 只在 `experiment.name` 上不同
- **THEN** B 的训练、科学评估与性能评估 Stage SHALL 复用 A 的输出

#### Scenario: 来源 Artifact 被修改
- **WHEN** 唯一复用键匹配的来源训练 Stage 的 ModelCheckpoint 文件 checksum 不再匹配
- **THEN** 系统 MUST NOT 复用该输出，SHALL 重新训练
