# governance/configuration-registry Specification

## Purpose
建立统一的配置字段路径、实现注册和构造前校验机制，使各 section 可以通过声明式选项组装，并在任何 Run、数据、模型或 Artifact 创建之前拒绝重复、未知和未决配置。

## Requirements

### Requirement: 配置字段路径统一映射到 Registry
项目 SHALL 在一个集中声明表中记录可注册字段的完整路径及其 Registry，至少覆盖 code、circuit、noise、generator、model、optimizer、scheduler、loss、trainer、decoder 和 data pipeline；业务组装代码 MUST NOT 通过散落的第三方名称条件分支选择实现。

#### Scenario: 按配置构造 generator
- **WHEN** `dataset.generator` 包含已注册 key
- **THEN** 组装器 SHALL 从 `dataset.generator` 对应 Registry 构造对象，而无需识别 Stim 或 CUDA-Q 类名

### Requirement: Registry 拒绝重复和未知 key
Registry SHALL 拒绝同名重复注册；构造未知 key 时 SHALL 在调用工厂前失败，并 SHALL 报告 Registry 名称、未知 key 和已登记选项。

#### Scenario: 重复注册
- **WHEN** 第二个工厂尝试以相同 key 注册到同一 Registry
- **THEN** 注册 SHALL 立即失败，原注册 MUST 保持不变

#### Scenario: 构造未知选项
- **WHEN** 配置请求 Registry 中不存在的 key
- **THEN** 构造 SHALL 失败并列出当前可选 key，不得静默选择 default 或 fallback

### Requirement: unresolved 在构造前 fail fast
配置预检 SHALL 递归发现 mapping 和 sequence 中值为字符串 `unresolved` 的路径；除非路径被调用方显式列入 allow set，否则验证 MUST 在任何 Run、数据、模型或 Artifact 创建前失败。

#### Scenario: generator version 未决
- **WHEN** `dataset.generator_version` 为 `unresolved` 且未被 allow
- **THEN** 预检 SHALL 报告完整路径并阻止后续工厂调用

#### Scenario: 阶段性允许模型字段未决
- **WHEN** 当前阶段只构建数据且调用方显式 allow `model.family`
- **THEN** 预检 SHALL 忽略该精确路径，但仍报告其他未被允许的 unresolved 路径

### Requirement: contracts-only 技术不得注册假实现
只有能够执行其 contract 的 adapter/runtime 才能注册为可构造选项；Protocol、technology catalog 条目或返回占位数据的函数 MUST NOT 注册到生产 Registry。

#### Scenario: Stim 只有接口定义
- **WHEN** Stim generator 尚未实现
- **THEN** `stim-syndrome-cpu` SHALL NOT 出现在可构造 Registry 中，选择它时预检 SHALL 明确失败

### Requirement: 配置网格只覆盖已声明字段
配置网格展开 SHALL 从一个基础配置按各轴取值的笛卡尔积生成相互独立的配置，并 SHALL 支持所有点共享的覆盖与按某个轴取值耦合的覆盖。每个覆盖 MUST 指向基础配置中已存在的完整字段路径；同一字段被多个来源覆盖、耦合覆盖缺少某个轴取值，或生成的实验名重复时，展开 MUST 在返回任何配置前失败。展开 MUST NOT 修改基础配置。

#### Scenario: 覆盖路径拼写错误
- **WHEN** 网格覆盖 `training.epoch`，而基础配置中只有 `training.epochs`
- **THEN** 展开 SHALL 报告完整路径 `training.epoch` 并失败，且不返回任何配置

#### Scenario: 展开论文网格
- **WHEN** 以 L ∈ {4, 6} 与 11 个 p 取值展开，并按 L 耦合隐藏单元数与 epoch 数
- **THEN** 展开 SHALL 返回 22 个实验名互不相同的配置，每个配置只在声明的字段上与基础配置不同
