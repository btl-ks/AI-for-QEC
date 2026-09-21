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
