# Spec Delta

## MODIFIED Requirements

### Requirement: TrainingSpec 与 ExecutionSpec 分离
TrainingSpec SHALL 描述 optimizer、learning rate、epoch、batch、scheduler 和 loss 等学习语义；ExecutionSpec SHALL 描述 device、CPU/GPU 资源、DDP、workers、AMP、compile 和训练步 executor 等运行语义。只改变训练步 executor MUST NOT 改变 DatasetKey，但 SHALL 改变完整 Experiment 身份与训练 Stage 复用键。

#### Scenario: 从单 GPU 切换到 DDP
- **WHEN** 用户保持 TrainingSpec 不变并只修改 ExecutionSpec 使用多个 GPU
- **THEN** DatasetKey MUST 保持不变，解析后的执行计划 SHALL 记录新的资源配置

#### Scenario: 从 eager 切换到 CUDA Graph
- **WHEN** 两次配置只有 `execution.step_executor` 不同
- **THEN** 两次配置 SHALL 解析到同一 DatasetKey，但 MUST 拥有不同 Experiment 身份，且训练 Stage MUST NOT 互相复用

## ADDED Requirements

### Requirement: 训练步 executor 经 Registry 选择且与模型结构解耦
`execution.step_executor` SHALL 通过 `REGISTRIES_BY_PATH` 对应 Registry 构造；可执行 executor SHALL 只依赖统一的 model、objective、optimizer、随机流与 Tensor minibatch contract，MUST NOT 按 RBM、CNN、RNN、GNN、Transformer 或其他模型家族名称在 trainer 或 orchestration 中分支选择行为。Protocol、catalog 条目或占位实现 MUST NOT 注册为可执行 executor。

#### Scenario: 选择已注册 eager executor
- **WHEN** 配置选择 `pytorch-eager`
- **THEN** resolved execution plan SHALL 记录该 executor，trainer SHALL 通过统一训练步 contract 执行现有 eager 更新

#### Scenario: 选择未知训练步 executor
- **WHEN** `execution.step_executor` 是未知 key 或字符串 `unresolved`
- **THEN** 配置预检 SHALL 在创建 Run、数据、模型或 Artifact 前失败，并报告完整字段路径与可用实现

#### Scenario: 不同模型使用同一 executor contract
- **WHEN** 两个 ModelFamily 提供同一训练步 contract 所需的 Tensor forward 与 objective，但内部结构分别属于不同模型家族
- **THEN** trainer SHALL 在不识别模型家族名称的情况下把二者交给同一个已选 executor

### Requirement: CUDA Graph executor 保持 minibatch 训练语义
`pytorch-cuda-graph` SHALL 将每个已声明训练 minibatch 恰好映射为一次 loss、gradient 与 optimizer 更新。为初始化、warmup 或 capture 执行的真实训练运算必须计入该 minibatch 的正常更新，MUST NOT 额外增加或减少 optimizer update、global step、样本消费或训练随机流消费；scheduler、epoch 监控和 checkpoint 边界 SHALL 与 eager 路径保持相同训练语义。

#### Scenario: 首次建立 graph
- **WHEN** executor 为某个受支持 batch signature 执行 warmup 和 capture
- **THEN** 被 warmup 或 capture 执行的每个 minibatch SHALL 各自产生且只产生一次已配置训练更新，训练记录中的 global step SHALL 等于已消费 minibatch 数

#### Scenario: 完成一个 epoch
- **WHEN** CUDA Graph executor 完成包含 N 个 minibatch 的 epoch
- **THEN** optimizer SHALL 恰好更新 N 次，scheduler SHALL 按既有 epoch 语义推进一次，并 SHALL 产生既有监控记录和 TrainingRecoveryCheckpoint

### Requirement: CUDA Graph 只保留 batch 级固定输入
CUDA Graph executor SHALL 继续消费现有 data pipeline 逐 minibatch 产生的目标设备 Tensor，并在 replay 前把当前 minibatch 复制到由 executor 持有的固定地址输入缓冲。持久化 DatasetArtifact、DatasetKey 与 DataLoader transfer contract MUST 保持不变；executor MUST NOT 要求或创建完整训练 split 的设备副本。

#### Scenario: replay 使用新的 minibatch
- **WHEN** 两个连续 minibatch 具有相同 signature 但内容不同
- **THEN** 第二次 replay SHALL 使用第二个 minibatch 的内容，且固定输入缓冲的容量 SHALL 只对应一个该 signature 的 minibatch

#### Scenario: 训练数据仍逐批传输
- **WHEN** 数据由 CPU DataLoader 送入 CUDA trainer
- **THEN** 现有 data pipeline SHALL 继续逐批完成 host-to-device transfer，CUDA Graph executor MUST NOT 把完整训练 split 预加载或缓存到 GPU

### Requirement: CUDA Graph 按有界 batch signature 捕获
batch signature SHALL 至少包含输入字段集合、各字段 shape、dtype 与 device。相同 signature SHALL 复用已经捕获的 graph；新 signature 只有在配置的 graph 数量上限内且 capture 兼容时才能建立独立 graph。executor MUST NOT 为满足已有 graph 而静默 padding、裁剪、丢弃或重排样本。

#### Scenario: 相同 signature 重放
- **WHEN** 当前 minibatch signature 已有成功捕获的 graph
- **THEN** executor SHALL 更新固定输入并 replay 该 graph，而不是重新 capture

#### Scenario: 遇到受支持的新 signature
- **WHEN** 当前 minibatch signature 尚未出现且未超过配置的 graph 上限
- **THEN** executor SHALL 以不增加额外训练更新的方式完成该 signature 的 warmup 与 capture，并在后续相同 signature 上复用

#### Scenario: 超过 signature 上限
- **WHEN** 新 minibatch signature 会使 graph 数超过配置上限
- **THEN** training Stage SHALL 明确失败并记录该 signature，MUST NOT 丢弃 minibatch、改变其 shape 或切换到 eager

#### Scenario: 模型含 capture 不兼容行为
- **WHEN** 模型或 objective 包含数据依赖控制流、CPU 运算或其他不能安全 capture 的行为
- **THEN** training Stage SHALL 明确失败并保留原异常上下文，MUST NOT 按模型名称选择替代实现或静默 fallback

### Requirement: CUDA Graph 随机流与恢复可审计
CUDA Graph executor SHALL 使用可安全 capture 的具名训练随机流；每次 replay SHALL 推进随机状态，MUST NOT 重复 capture 时的随机样本。epoch checkpoint SHALL 保存 replay 后的模型、optimizer、scheduler、shuffle 与训练随机流状态；恢复 Attempt SHALL 从已验证 checkpoint 恢复这些状态并重新建立进程内 graph，MUST NOT 序列化 CUDA Graph 对象或因重新 capture 增加训练步骤和随机流消费。

#### Scenario: 连续 replay 消费随机数
- **WHEN** 同一 graph 连续 replay 两次且 objective 使用训练随机流
- **THEN** 两次更新 SHALL 使用顺序推进的随机状态，而不是重复 capture 时的随机输出

#### Scenario: 从 epoch checkpoint 恢复 CUDA Graph 训练
- **WHEN** 新 Attempt 从 CUDA Graph 训练产生的有效 epoch-k TrainingRecoveryCheckpoint 恢复
- **THEN** 它 SHALL 从 epoch k+1 继续，重新 capture 所需 graph 而不持久化或复用旧进程的 graph 对象；在相同且声明为确定的执行环境中，最终训练状态 SHALL 与不中断的同 executor 训练一致

#### Scenario: ModelCheckpoint 用于推理
- **WHEN** CUDA Graph 训练完成并提交 ModelCheckpoint
- **THEN** ModelCheckpoint SHALL 只包含既有推理模型 payload，MUST NOT 包含 optimizer、训练随机流或 CUDA Graph runtime 对象

### Requirement: CUDA Graph 执行必须留证且不得静默 fallback
Training Stage SHALL 记录请求和实际训练步 executor、executor 版本、device、batch signatures、warmup 步数、capture 次数与耗时、replay 次数以及是否发生 fallback。选择 `pytorch-cuda-graph` 后，任何 capture 或 replay 失败 MUST 使 Stage 失败；成功 Stage 的 `fallback_observed` MUST 为 false，且只有实际完成至少一次 capture 与 replay 时才能声明 CUDA Graph 已执行。

#### Scenario: CUDA Graph 成功执行
- **WHEN** `pytorch-cuda-graph` 完成训练 Stage
- **THEN** Stage evidence SHALL 表明 requested 与 observed executor 均为 `pytorch-cuda-graph`、`fallback_observed=false`，并包含非零 capture 与 replay 计数

#### Scenario: capture 失败
- **WHEN** CUDA Graph capture 抛出错误
- **THEN** Attempt SHALL 把 training Stage 标记为失败或中断并记录错误，MUST NOT 生成伪造的成功 ModelCheckpoint 或继续以 eager 训练

#### Scenario: CUDA Graph 模型进入科学评估
- **WHEN** CUDA Graph 训练产生的新 ModelCheckpoint 用于论文工作流
- **THEN** 工作流 SHALL 重新运行 Scientific Evaluation 与 Accuracy Gate，MUST NOT 因 eager 模型具有相同架构而复用其训练或科学评估结果
