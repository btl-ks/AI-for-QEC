# Proposal

## Why

当前工作区只有架构说明文档，缺少可执行的仓库结构、正式需求源与稳定的公共接口，因而无法安全地开展后续实现或验证。现在需要先建立一个最小、可导入、可校验的契约仓库，并明确所有领域能力仍处于“仅接口、未实现”状态。

## What Changes

- 初始化 Python 项目、版本控制基础文件、公共包结构与契约测试。
- 建立 OpenSpec 需求、变更、任务和验证工作流，并登记 capability 状态。
- 定义 QEC、噪声、数据集、实验生命周期、训练执行、解码和科学评估的稳定类型与 `Protocol` 接口。
- 通过 `ai_qec.notebook_api` 统一导出公共契约。
- 明确不实现数据生成、训练、解码、恢复、缓存、性能优化或硬件后端。

## Capabilities

### New Capabilities

- `governance/spec-management`: 定义 OpenSpec 作为正式需求、变更、验收与证据的唯一真相源。
- `research/experiment-lifecycle`: 定义 Experiment、Attempt/Run、Stage、Artifact、随机流与恢复边界。
- `qec/dataset-pipeline`: 定义 QEC/Noise、DatasetSpec/Artifact/Instance、QECBatch、注册与解析接口。
- `qec/decoder-workbench`: 定义统一解码请求、结果和 AI/经典解码器接口。
- `research/training-contracts`: 分离模型训练语义与执行资源语义，并区分模型 checkpoint 与恢复 checkpoint。
- `research/scientific-evaluation`: 定义同协议科学评估、LER 估计和 Accuracy Gate 契约。

### Modified Capabilities

无。

## Impact

- 新增 `ai_qec` 公共 Python 包，但只包含值对象、枚举和抽象协议。
- 新增 `openspec/` 正式需求结构、capability 清单及验证材料。
- 新增最小配置样例与契约级测试。
- 不增加运行时第三方依赖，不声称任何研究或执行能力已经实现。
