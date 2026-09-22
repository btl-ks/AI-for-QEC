# research/training-contracts Specification

## Purpose
分离影响学习问题的训练语义与只影响运行方式的执行资源语义，并明确模型产物和故障恢复状态的不同生命周期与消费用途。

## Requirements

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

### Requirement: v0.1 Trainer 使用 PyTorch CUDA
v0.1 SHALL 选择 PyTorch 作为 trainer framework，并在 GPU 执行时使用 CUDA；trainer SHALL 接收 Tensor-first QECBatch，同时保留 CPU 执行作为可测试的执行配置而非独立训练语义。

#### Scenario: 解析 CUDA 训练配置
- **WHEN** ExecutionSpec 请求单 GPU CUDA trainer
- **THEN** resolved plan SHALL 记录 `pytorch` framework、CUDA device、worker 数、mixed precision 和 compile 设置

### Requirement: Trainer 不依赖生成后端
PyTorch trainer SHALL 只消费符合 QECBatch contract 的 Tensor 表示，不得要求模型代码识别 Stim 或 CUDA-Q；数据驻留与 transfer policy SHALL 由 data pipeline 负责。

#### Scenario: 在 CPU 与 GPU generator 间切换
- **WHEN** 相同数据语义从 Stim CPU generator 切换到 CUDA-Q GPU generator
- **THEN** trainer 接口 SHALL 保持不变，只有 batch layout 与 transfer path 可以改变

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
