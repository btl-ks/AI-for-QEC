# Spec Delta

## ADDED Requirements

### Requirement: 联合 RBM 的 CD-k 训练
PyTorch trainer SHALL 以 `[e | S]` 联合可见层训练 RBM，每个 epoch 用 training_shuffle 随机流打乱训练集，并对每个 minibatch 执行一次 CD-k 梯度更新；每个 epoch 结束时 SHALL 记录训练与验证监控指标，并 SHALL 写出一个 TrainingRecoveryCheckpoint；训练结束时 SHALL 写出一个只含模型状态的 ModelCheckpoint。

#### Scenario: 完成训练
- **WHEN** 训练在配置的 epoch 数内正常结束
- **THEN** 系统 SHALL 产生每个 epoch 一个 TrainingRecoveryCheckpoint 和一个 ModelCheckpoint，两者 checksum 可校验且类型不可互换

### Requirement: epoch 边界恢复保持训练轨迹
从经过校验的 TrainingRecoveryCheckpoint 恢复时，trainer SHALL 恢复模型、optimizer、scheduler、随机流状态和已完成 epoch 的监控历史，并从下一个 epoch 继续；在确定性设备上，恢复后的最终模型 SHALL 与未中断训练得到的模型逐位相同。

#### Scenario: 训练中断后恢复
- **WHEN** 一个 Attempt 在第 k 个 epoch 结束后被中断，新的恢复 Attempt 使用其 epoch-k checkpoint
- **THEN** 恢复 Attempt SHALL 只训练剩余 epoch，且 CPU 上的最终权重 SHALL 与一次性训练完全相同

### Requirement: 本地 runtime 拒绝不支持的执行选项
本地 runtime SHALL 只支持单进程、单设备 float32 训练；请求 `distributed`、mixed precision、`compile_model`、多于一个 GPU，或请求不可用的 CUDA 设备时，系统 MUST 在创建 Run 前失败，MUST NOT 静默回退到 CPU 或其他执行方式。

#### Scenario: CUDA 不可用
- **WHEN** ExecutionSpec 请求 `cuda` 但当前环境没有可用 CUDA 设备
- **THEN** 创建 Experiment SHALL 失败并报告请求的 device，且 MUST NOT 创建 Run 目录
