# Proposal

## Why

项目规范已经要求归档前执行 OpenSpec verify，但 Zephy 的全局 OpenSpec `core` profile 未启用 `verify` workflow，当前项目也尚未生成本地 workflow 文件。需要补齐工具配置，使规范要求具有可直接调用的工作流实现。

## What Changes

- 将 OpenSpec 全局 profile 从 `core` 调整为 `custom`。
- 保留 `propose`、`explore`、`apply`、`update`、`sync`、`archive`，新增 `verify`。
- 保持 `delivery: both`；各 Agent 工具按自身支持形式物化，Codex 只生成 skills。
- 在当前 QEC 项目配置 Codex 目标并应用 profile，生成所选 workflows。
- 不修改 QEC 功能规格、Python 代码、公共 API 或 capability 状态。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

无；这是纯工具配置变更，`.openspec.yaml` 使用 `skip_specs: true`。

## Impact

- 全局配置：`/home/zephy/.config/openspec/config.json`
- 当前项目生成的 Agent workflow 文件
- 不影响研究运行时、数据集或模型结果
