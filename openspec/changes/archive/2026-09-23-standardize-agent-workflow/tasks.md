# Tasks

## 1. 规范实施

- [x] 1.1 按“Agent 工作范围由用户意图约束”“可实施 change 保持需求到证据追踪”和“OpenSpec 交付门禁区分规划校验与实现验证”的全部场景，以及 design 决策 1–3，扩充项目 `AGENTS.md`；用 `rg -n "Work Scope by User Intent|Requirement-to-Evidence Chain|Change Delivery Gates|Archive Blockers" AGENTS.md` 验证四个规范章节存在。
- [x] 1.2 按“共享与项目 Agent 规范分层”的两个场景及 design 决策 1、2、4，更新 `/home/zephy/.codex/AGENTS.md`，保留既有文本格式规则并加入条件式 OpenSpec 规则；用 `rg -n "用户意图|OpenSpec 项目|不得自动初始化|无需.*apply" /home/zephy/.codex/AGENTS.md` 验证关键约束存在。

## 2. 一致性与证据

- [x] 2.1 对照 delta spec 和 design 检查项目级与 Zephy 级规范语义一致且非 OpenSpec 项目不被强制初始化；运行 `openspec validate standardize-agent-workflow --strict --no-interactive` 和 `openspec validate --all --strict --no-interactive` 并确认全部通过。
- [x] 2.2 补齐 `openspec/templates/verification.md` 并据此创建本 change 的 `verification.md`，记录规范文本检查、OpenSpec validation、外部共享规范摘要与限制；验证证据完整后确认 change 可同步 `governance/spec-management` 主规格并归档。
