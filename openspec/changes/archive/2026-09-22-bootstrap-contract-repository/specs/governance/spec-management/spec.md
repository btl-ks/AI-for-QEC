# Spec Delta

## Purpose

建立项目正式需求、受控变更、验收证据与能力状态之间的唯一治理路径，防止架构意图、接口骨架或占位文件被误报为已经实现的研究能力。

## ADDED Requirements

### Requirement: OpenSpec 是正式需求源
项目 SHALL 将 `openspec/specs/` 与尚未归档的 `openspec/changes/` 作为行为需求、变更范围和验收条件的唯一正式真相源；架构说明 SHALL 仅作为设计背景。

#### Scenario: 架构说明与正式规格冲突
- **WHEN** 架构说明与已归档规格描述不同的行为
- **THEN** 维护者 MUST 以已归档规格为准，并通过新的 OpenSpec change 提议行为变更

### Requirement: 行为变更使用受控流程
任何新增或修改公共行为的工作 SHALL 在编码前拥有有界的 proposal、spec delta、design 和 tasks，并在归档前通过严格验证。

#### Scenario: 开始实现公共行为
- **WHEN** 维护者准备实现或修改一个公共 contract
- **THEN** 对应 change MUST 明确该行为、验收场景、设计边界和实施任务

### Requirement: 能力声明以证据为依据
Capability 状态 SHALL 区分计划、可用和已验证；仅存在接口、目录或占位文件 MUST NOT 被视为能力已经可用。

#### Scenario: 只有接口骨架
- **WHEN** capability 只有类型和 Protocol 定义而没有执行实现
- **THEN** capability SHALL 保持 `planned`，且清单 MUST 明确记录“contracts only”
