# Spec Delta

## ADDED Requirements

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
