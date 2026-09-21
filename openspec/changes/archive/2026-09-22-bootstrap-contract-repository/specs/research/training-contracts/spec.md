# Spec Delta

## Purpose

分离影响学习问题的训练语义与只影响运行方式的执行资源语义，并明确模型产物和故障恢复状态的不同生命周期与消费用途。

## ADDED Requirements

### Requirement: TrainingSpec 与 ExecutionSpec 分离
TrainingSpec SHALL 描述 optimizer、learning rate、epoch、batch、scheduler 和 loss 等学习语义；ExecutionSpec SHALL 描述 device、CPU/GPU 资源、DDP、workers、AMP 和 compile 等运行语义。

#### Scenario: 从单 GPU 切换到 DDP
- **WHEN** 用户保持 TrainingSpec 不变并只修改 ExecutionSpec 使用多个 GPU
- **THEN** DatasetKey MUST 保持不变，解析后的执行计划 SHALL 记录新的资源配置

### Requirement: 不支持的执行选项必须提前失败
执行请求的组合不受支持时，validator SHALL 在创建 Run、生成数据或写入 Artifact 之前返回明确错误。

#### Scenario: 请求无效设备组合
- **WHEN** ExecutionSpec 声明互相冲突或环境不支持的选项
- **THEN** 系统 SHALL 在产生持久化副作用前拒绝该配置

### Requirement: 模型 checkpoint 与恢复 checkpoint 分离
ModelCheckpoint SHALL 仅用于推理、评估、部署或科学复用；TrainingRecoveryCheckpoint SHALL 可额外包含 optimizer、scheduler、AMP scaler、epoch/step、RNG 和恢复 cursor，且 MUST NOT 被冒充为普通模型产物。

#### Scenario: 从 epoch 边界恢复训练
- **WHEN** 新 Attempt 从经过验证的 TrainingRecoveryCheckpoint 恢复
- **THEN** 恢复接口 SHALL 能识别恢复来源和 checkpoint 类型，并拒绝把仅含模型权重的产物当作完整恢复状态
