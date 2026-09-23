# Design

## Context

OpenSpec 1.13.1 的 `core` profile 默认启用六个 workflows，不包含可选的 `verify`。全局配置决定选择哪些 workflows，`openspec update` 负责把配置物化到具体项目的已检测 Agent 工具目录。

## Goals / Non-Goals

**Goals:**

- 以 OpenSpec 官方配置命令启用 `verify`。
- 保持现有 core workflows 和 `delivery: both` 不变。
- 在 QEC 项目为 Codex 生成并检查实际 workflow skill 文件。

**Non-Goals:**

- 不直接复制旧项目的生成文件。
- 不修改 OpenSpec 安装包或 QEC 功能实现。
- 不自动更新其他项目；其他项目可单独运行 `openspec update`。

## Decisions

### 1. 使用官方交互式 profile 配置

运行 `openspec config profile`，选择 workflows only，保留六个 core workflows 并加入 `verify`。OpenSpec 自动把 profile 标记为 `custom`，避免直接编辑全局 JSON。

### 2. 配置 Codex 目标并应用到当前项目

在交互确认中选择立即应用；若项目尚未配置 Agent 工具，则使用 `openspec init --tools codex --profile custom` 只补充 Codex 工具目标，再运行 `openspec update` 检查可重复同步。全局 `delivery: both` 保持不变，但 Codex 官方目标只使用 skills，因此 slash commands 会被 OpenSpec 明确跳过。完成后检查生成的七个 skills 是否同时包含 `verify` 与既有 workflows。

### 3. 通过配置与生成物双重验证

`openspec config list` 验证全局选择；项目生成文件和 `openspec doctor --json` 验证当前项目同步结果；严格 OpenSpec validation 验证 change 本身。

### 4. 生成型 Agent 状态不进入项目源码

`.agents/` 由本机 OpenSpec profile 和工具目标生成，加入 `.gitignore`。项目只版本化 OpenSpec change、验证证据与治理规范；生成 skills 通过文件清单和 SHA-256 记录身份，需要时用 `openspec update` 重建。

## Risks / Trade-offs

- [全局 profile 影响后续项目更新] → 只增加 `verify`，保留所有现有选择；其他项目只有运行 `openspec update` 后才物化。
- [生成文件数量增加] → 文件由 OpenSpec 1.13.1 管理，不手工修改；后续用 `openspec update` 刷新。
