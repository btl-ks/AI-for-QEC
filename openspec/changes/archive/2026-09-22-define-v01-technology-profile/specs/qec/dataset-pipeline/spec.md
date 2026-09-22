# Spec Delta

## ADDED Requirements

### Requirement: v0.1 QEC 后端矩阵
v0.1 SHALL 选择 Qiskit 作为 circuit adapter、Stim 作为 CPU QEC 与 Stim-noise adapter、CUDA-Q 作为 GPU QEC adapter；syndrome generation SHALL 支持 Stim CPU 路径和 CUDA-Q GPU 路径。缺少选定 adapter 时系统 MUST 明确报告 capability unavailable，且 MUST NOT 静默切换后端。

#### Scenario: 请求尚未实现的 CUDA-Q generator
- **WHEN** 用户选择 CUDA-Q GPU syndrome generator 但 adapter 尚未实现或环境不兼容
- **THEN** 系统 SHALL 在生成样本前失败并报告缺失的 technology/capability ID

### Requirement: QECBatch 采用 Tensor-first 运行时
AI pipeline 的主要运行时表示 SHALL 是 PyTorch Tensor；contract SHALL 同时能够描述 NumPy、bit-packed CPU buffer、PyTorch CUDA Tensor 和 DLPack 交换边界。任何表示转换 MUST 保留 sample identity、dataset identity、shape、dtype、device 和 provenance。

#### Scenario: 持久化数据进入训练
- **WHEN** bit-packed CPU 数据被加载到 PyTorch trainer
- **THEN** pipeline SHALL 显式记录从持久化表示到 PyTorch Tensor 的转换，并保持 QECBatch 身份字段不变

### Requirement: CPU 到 GPU 使用 PyTorch 标准流水线
v0.1 CPU→GPU 路径 SHALL 使用 PyTorch Dataset/DataLoader，并支持 pinned memory、prefetch 与 non-blocking host-to-device transfer；DALI 和 Ray Data MUST NOT 成为 v0.1 的必需依赖。

#### Scenario: Stim CPU batch 送入 CUDA trainer
- **WHEN** Stim 在 CPU 产生 batch 且目标 trainer 位于 CUDA device
- **THEN** pipeline SHALL 能声明 pinned-memory 与 non-blocking transfer policy，并将最终 CUDA device 记录在 batch layout 中

### Requirement: GPU 到 GPU 优先零拷贝
CUDA-Q→PyTorch CUDA 路径 SHALL 优先保持数据驻留 GPU，并 SHALL 优先使用兼容的 CUDA Tensor 或 DLPack 边界；GPU→CPU→GPU host staging MUST NOT 静默发生。

#### Scenario: 零拷贝不可用
- **WHEN** CUDA-Q 输出无法与 PyTorch 通过声明的 device/stream/interchange contract 安全共享
- **THEN** pipeline MUST 拒绝该路径或显式记录 host-staging fallback，且不得把 fallback 报告为 zero-copy

