# research/experiment-lifecycle Specification

## Purpose
定义可审计的 Experiment、Attempt、Stage、Artifact、随机流和恢复语义，使 Notebook 或脚本在中断后能够创建新尝试并从经过验证的安全边界继续研究流程。

## Requirements

### Requirement: Experiment 包含不可变历史 Attempt
一个 Experiment SHALL 允许包含多个 Attempt；进入终态的 Attempt MUST 保持其 `COMPLETED`、`FAILED`、`INTERRUPTED` 或 `PARTIAL` 状态，不得原地复活。

#### Scenario: 失败后发起恢复
- **WHEN** 一个 Attempt 在训练阶段失败并请求恢复
- **THEN** 系统 SHALL 保留原 Attempt 的失败状态并创建带有 `recovery_from` 引用的新 Attempt

### Requirement: Stage 记录可验证边界
每个 Stage SHALL 记录身份、状态、时间、输入、输出、Artifact 引用、错误和恢复元数据；v0.1 恢复单位 SHALL 是 Stage 边界而不是 Notebook cell。

#### Scenario: Notebook 内核重启
- **WHEN** 用户在干净内核中重新执行 Notebook
- **THEN** 生命周期接口 SHALL 允许检查先前 Stage 记录并只复用校验成功的已完成输出

### Requirement: Artifact 不可变且可校验
已提交 Artifact SHALL 拥有稳定身份、生产 Attempt、生产 Stage、位置、checksum 和 metadata；已登记内容发生变化时完整性验证 MUST 失败。

#### Scenario: 已登记 Artifact 被修改
- **WHEN** Artifact 的实际内容与登记 checksum 不一致
- **THEN** 消费方 MUST 拒绝该 Artifact，且不得把它标记为可安全复用

### Requirement: 随机流具名且独立
随机性 SHALL 通过具名 stream 表达；相同 master seed 与 stream name SHALL 可重建相同序列，消费一个 stream MUST NOT 改变其他 stream 的序列。

#### Scenario: 训练随机性不影响数据抽样
- **WHEN** 两次实验只改变 `training_shuffle` 的消费次数
- **THEN** `qec_sampling` 和 `dataset_split` stream SHALL 产生与原实验相同的序列

### Requirement: start_or_recover 创建新 Attempt
Experiment 身份 SHALL 由完整规范化配置的摘要决定；`start_or_recover()` 在没有历史 Attempt 时 SHALL 创建全新 Attempt，在存在终态 Attempt 时 SHALL 创建引用该 Attempt 的新 Attempt，且 MUST NOT 修改任何终态 Attempt 的状态。

#### Scenario: 相同配置再次执行
- **WHEN** 同一配置的上一个 Attempt 已处于 `completed` 或 `failed`
- **THEN** 系统 SHALL 创建编号递增的新 Attempt，其 `recovery_from` SHALL 引用上一个 Attempt 与其终态

### Requirement: 陈旧 RUNNING Attempt 显式中断
当最新 Attempt 仍为 `running` 但其所属进程已不存在，或同一进程重新开始同一 Experiment 时，系统 SHALL 先把它标记为 `interrupted` 并记录原因，再创建恢复 Attempt；所属进程仍在运行且不是当前进程时，系统 MUST 拒绝启动。

#### Scenario: 内核被强制终止后重跑
- **WHEN** 上一个 Attempt 的进程在训练中被杀死，用户在新内核中重新执行 Notebook
- **THEN** 旧 Attempt SHALL 变为 `interrupted`，新 Attempt SHALL 引用它恢复

### Requirement: 只复用校验通过的已完成 Stage
恢复 Attempt SHALL 只复用源 Attempt 中状态为 `completed` 且全部输出 Artifact checksum 校验通过的 Stage；训练 Stage 未完成时 SHALL 从源 Attempt 链上 epoch 最大且校验通过的 TrainingRecoveryCheckpoint 继续。复用关系 SHALL 记录在新 Attempt 的 Stage 记录中。

#### Scenario: 源 Artifact 被修改
- **WHEN** 源 Attempt 已完成训练 Stage，但其 ModelCheckpoint 文件 checksum 不再匹配
- **THEN** 恢复 Attempt MUST NOT 复用该训练输出，SHALL 改为从校验通过的恢复 checkpoint 或从头训练，并记录原因

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
