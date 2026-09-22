# Spec Delta

## Purpose

建立机器可读、可验证且与 capability 状态分离的技术选型目录，使项目能够明确 v0.1 的 primary 与 alternative 工具，同时不会把选型决定误报为 adapter 已实现。

## ADDED Requirements

### Requirement: 技术选型目录是正式配置
项目 SHALL 使用受 schema 约束的 technology catalog 记录每个组件的 primary technology、alternatives、目标阶段和 implementation status；技术选型变更 MUST 通过 OpenSpec change。

#### Scenario: 查询 v0.1 技术基线
- **WHEN** 维护者读取 technology catalog
- **THEN** 每个受管组件 SHALL 有唯一 ID、唯一 primary、明确 alternatives、phase 和 implementation status

### Requirement: 技术选择不等于实现
目录中的 `selected` 或 `primary` SHALL 只表示项目决策；只有 adapter/runtime 实现、测试与 verification evidence 完整时，implementation status 才能标记为 implemented。

#### Scenario: 只有 Protocol 和目录条目
- **WHEN** 一个技术条目只有接口定义而没有第三方 adapter 实现
- **THEN** 条目 MUST 保持 `contracts-only`，对应执行 capability MUST 保持 `planned`

### Requirement: 核心 contract 与技术 adapter 分离
QECSpec、NoiseSpec、QECBatch、DatasetArtifact、DecodeRequest 和 DecodeResult SHALL 保持 vendor-neutral；Qiskit、Stim、CUDA-Q、PyTorch 和 PyMatching MUST 通过 adapter/runtime 边界接入。

#### Scenario: 替换选定技术
- **WHEN** 后续 change 替换一个 primary technology
- **THEN** 核心 contract SHALL 无需因第三方对象类型而改变

