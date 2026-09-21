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
