# qec/dataset-pipeline Specification

## Purpose
定义与生成后端解耦的 QEC 数据契约、可复用不可变数据集及其解析规则，使不同模型、训练配置和执行资源能够安全共享同一份科学数据。

## Requirements

### Requirement: 数据集采用三层模型
平台 SHALL 区分描述期望数据的 `DatasetSpec`、表示不可变物理数据的 `DatasetArtifact` 和表示某个 Run 使用关系的 `DatasetInstance`。

#### Scenario: 两个 Run 使用相同数据
- **WHEN** 两个 Run 解析到同一个经过验证的 DatasetArtifact
- **THEN** 每个 Run SHALL 拥有独立 DatasetInstance，且两者 SHALL 引用同一个 DatasetArtifact 身份

### Requirement: DatasetKey 只包含数据生成语义
DatasetKey SHALL 包含 QEC、Noise、generator 及版本、具有科学意义的 backend 语义、样本数、seed、split、影响持久化数据的 preprocessing 和 schema；模型、训练超参数与执行资源 MUST NOT 参与身份计算。

#### Scenario: 只改变 GPU 数量
- **WHEN** 两次实验只有 ExecutionSpec 的 GPU 数量不同
- **THEN** 两次实验 SHALL 解析到相同 DatasetKey

### Requirement: 缓存命中必须先验证
Dataset resolver SHALL 在命中 registry 后验证 manifest 与内容完整性；缓存未命中时，新 Artifact SHALL 在验证通过后以不可变方式提交并登记。

#### Scenario: registry 指向损坏数据
- **WHEN** DatasetKey 命中但 Artifact checksum 验证失败
- **THEN** resolver MUST 拒绝缓存命中并返回明确的完整性错误

### Requirement: 后端输出统一 QECBatch
所有数据生成后端 SHALL 输出统一的 QECBatch 语义，至少表达 syndrome/detector events、observable truth、sample identity、dataset identity 和 provenance；模型 MUST NOT 依赖生成后端名称。

#### Scenario: 替换数据生成后端
- **WHEN** Stim 与另一个兼容后端表达相同 QEC 与 Noise 语义
- **THEN** 下游 trainer 和 decoder SHALL 通过同一 QECBatch contract 接收数据

### Requirement: 噪声近似必须显式
当后端不能精确表达 NoiseSpec 时，adapter MUST 拒绝请求或返回显式记录的 approximation；adapter MUST NOT 静默改变噪声语义。

#### Scenario: 后端不支持精确噪声
- **WHEN** 请求的噪声模型无法由所选后端精确表示且未授权近似
- **THEN** adapter SHALL 在生成任何数据前失败

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

### Requirement: Toric code-capacity 相位翻转数据生成
平台 SHALL 能够为 L×L 环面 toric code 在独立相位翻转 code-capacity 噪声下生成 train、validation、test 三个 split；每个样本 MUST 同时包含物理错误链、顶点 syndrome 和逻辑可观测量翻转，且各 split 使用由 dataset seed 派生的独立具名随机流。

#### Scenario: 生成论文数据集
- **WHEN** 配置选择 `toric` code、`independent-phase-flip` 噪声和 `stim-syndrome-cpu` generator
- **THEN** 解析得到的 DatasetArtifact SHALL 含有样本数与 DatasetSpec 一致的三个 split，且每个样本 SHALL 包含 2L² 位物理错误、L² 位 syndrome 和 2 位逻辑可观测量

#### Scenario: 请求 generator 不支持的语义
- **WHEN** 配置请求多轮、circuit-level、非 X 基逻辑或 generator 不支持的噪声
- **THEN** 系统 SHALL 在生成任何样本前失败并报告不支持的字段，且 MUST NOT 以近似语义生成数据

### Requirement: 生成后端版本必须精确匹配
`dataset.generator_version` SHALL 是一个明确版本；当安装的生成后端版本与之不一致时，系统 MUST 在创建 Run 或生成数据前失败，MUST NOT 静默使用另一版本。

#### Scenario: 已安装后端版本不同
- **WHEN** 配置声明的 generator 版本与运行环境中的后端版本不同
- **THEN** 预检 SHALL 失败并同时报告声明版本和实际版本

### Requirement: 提交前校验码语义一致性
新生成的 DatasetArtifact SHALL 在提交前用与生成后端无关的码定义校验每个样本的 syndrome 等于物理错误的边界、逻辑可观测量等于物理错误与逻辑算符的奇偶性；任何不一致 MUST 阻止提交。消费已提交 split 前，平台 SHALL 重新校验分片 checksum。

#### Scenario: 后端输出与码定义不一致
- **WHEN** 暂存数据中任一样本的 syndrome 与其物理错误不一致
- **THEN** 系统 MUST 拒绝提交并报告不一致的 split，registry SHALL 保持不变

#### Scenario: 已提交分片被修改
- **WHEN** 训练或评估读取的分片 checksum 与 manifest 记录不一致
- **THEN** 读取 SHALL 失败并报告完整性错误，且 MUST NOT 使用该分片
