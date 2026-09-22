# Spec Delta

## ADDED Requirements

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

