# Spec Delta

## ADDED Requirements

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
