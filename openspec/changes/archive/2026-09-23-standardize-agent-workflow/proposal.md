# Proposal

## Why

当前项目只记录了 OpenSpec 的线性交付顺序，没有明确“分析、设计、实现、完成、继续”等用户动词各自授权到哪一层，也没有固化 requirement-to-evidence 追踪链、验证门禁和归档阻断条件。这会让 Agent 在规划与实现之间产生歧义，并使项目规范与 Zephy 级共享规范难以保持一致。

## What Changes

- 在项目 `AGENTS.md` 中定义用户意图到工作范围的规范映射，并规定多个动作动词按显式范围取并集。
- 明确“实现/完成”一次性授权完整 OpenSpec 生命周期，不要求用户在规划完成后再次说 `apply`；“设计/计划”仍严格停在规划与校验边界。
- 固化 Requirement → Functional design → Implementation task → Test evidence 追踪链。
- 增加严格规划校验、实现、测试、OpenSpec verify、`verification.md`、spec 同步、capability 更新与归档的交付门禁及阻断条件。
- 将通用动词授权和 OpenSpec 项目门禁同步到 Zephy 级共享 Agent 规范；非 OpenSpec 项目不被强制初始化或采用 OpenSpec。
- 不改变 QEC 公共 API、实验运行时、数据格式或科学能力状态。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `governance/spec-management`：补充 Agent 用户意图授权、需求到证据追踪、验证与归档门禁的正式治理要求。

## Impact

- 项目规范：`AGENTS.md`
- Zephy 级共享规范：`/home/zephy/.codex/AGENTS.md`
- OpenSpec 正式规格：`openspec/specs/governance/spec-management/spec.md`
- 不新增运行时依赖，不修改 Python 实现或 Notebook API。
