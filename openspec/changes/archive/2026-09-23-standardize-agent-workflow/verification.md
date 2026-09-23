# Verification Evidence: standardize-agent-workflow

## Verification Identity

- Change：`standardize-agent-workflow`
- Schema：`spec-driven`
- 被测提交：`dc2c83f3726af0a13caefc4a9918e563ce0ee01d` 加本 change 记录的未提交文件摘要
- 分支：`main`
- 验证时间：`2026-09-23T19:22:11+09:00`
- 验证者：Codex

## Environment

- 操作系统与运行时：Linux 6.6.87.2-microsoft-standard-WSL2 x86_64；OpenSpec 1.13.1
- Python 与依赖锁身份：不适用；本 change 不修改 Python 或依赖
- 加速器、驱动与后端：不适用
- 数据集身份与 manifest 哈希：不适用
- 配置快照与哈希：项目 `openspec/config.yaml`；项目与 Zephy 级规范的 SHA-256 见 Evidence Integrity

## Scope

- 纳入的 requirements 与 scenarios：Agent 工作范围由用户意图约束（3 个场景）、可实施 change 保持需求到证据追踪（2 个场景）、OpenSpec 交付门禁区分规划校验与实现验证（2 个场景）、共享与项目 Agent 规范分层（2 个场景）
- 排除项及原因：未修改 QEC Python 实现、公共 API、数据与运行产物；未强迫非 OpenSpec 项目采用 OpenSpec；未修改 OpenSpec CLI 安装或全局 workflow profile
- OpenSpec task identities：`standardize-agent-workflow/1.1`、`1.2`、`2.1`、`2.2`

## Verification Summary

| 维度 | 实际结果 | 状态 |
| --- | --- | --- |
| Completeness | 4/4 tasks；4 个 requirements、9 个 scenarios 均有实现位置和证据 | PASS |
| Correctness | 项目与 Zephy 级关键文本断言通过；严格 change validation 与全仓 validation 通过 | PASS |
| Coherence | 两层规范职责分明；项目规则保持 QEC 特有约束；非 OpenSpec 项目明确不自动初始化 | PASS |

- CRITICAL 问题：0
- WARNING 问题：1
- SUGGESTION 问题：0

## Requirement-to-Evidence Matrix

| Requirement | Scenario | Design 章节 | Task | 实现位置 | 自动化测试或命令 | 结果 | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Agent 工作范围由用户意图约束 | 用户只要求设计 | 决策 1–2 | 1.1、1.2 | `AGENTS.md:26`；`/home/zephy/.codex/AGENTS.md:3` | `rg -n "Work Scope by User Intent" AGENTS.md /home/zephy/.codex/AGENTS.md` | PASS | 两层规范均限制设计请求不得自动实现 |
| Agent 工作范围由用户意图约束 | 用户要求完成实现 | 决策 2 | 1.1、1.2 | `AGENTS.md:33`；`/home/zephy/.codex/AGENTS.md:10` | `rg -n "无需.*apply|不得要求.*apply" AGENTS.md /home/zephy/.codex/AGENTS.md` | PASS | 两层规范均保持一次性实现授权 |
| Agent 工作范围由用户意图约束 | 用户要求分析并记录 | 决策 1–2 | 1.1、1.2 | `AGENTS.md:30`；`/home/zephy/.codex/AGENTS.md:7` | 人工核对动作并集段落 | PASS | `AGENTS.md:36`；共享规范第 13 行 |
| 可实施 change 保持需求到证据追踪 | 编写实施任务 | 决策 3 | 1.1 | `AGENTS.md:40` | `rg -n "Requirement-to-Evidence Chain|Implementation task|Test evidence" AGENTS.md` | PASS | 四级追踪链完整 |
| 可实施 change 保持需求到证据追踪 | 证据尚未通过 | 决策 3 | 1.1 | `AGENTS.md:47`；`AGENTS.md:59` | 人工核对 task 勾选门禁 | PASS | 证据通过后才允许完成任务 |
| OpenSpec 交付门禁区分规划校验与实现验证 | 规划严格校验通过 | 决策 3 | 1.1、2.1 | `AGENTS.md:51`；`AGENTS.md:64` | `openspec validate standardize-agent-workflow --strict --no-interactive` | PASS | change valid，但规范明确 validation 不代表 implementation |
| OpenSpec 交付门禁区分规划校验与实现验证 | 归档前存在阻断项 | 决策 3 | 1.1 | `AGENTS.md:78`；共享规范第 31 行 | `rg -n "Archive Blockers|不得归档" AGENTS.md /home/zephy/.codex/AGENTS.md` | PASS | 两层规则均列出归档阻断项 |
| 共享与项目 Agent 规范分层 | 非 OpenSpec 项目收到实现请求 | 决策 1 | 1.2 | `/home/zephy/.codex/AGENTS.md:17` | `rg -n "不得自动初始化 OpenSpec|不得擅自创建.*openspec" /home/zephy/.codex/AGENTS.md` | PASS | 未采用 OpenSpec 的项目不被强制初始化 |
| 共享与项目 Agent 规范分层 | QEC 项目收到行为变更请求 | 决策 1、4 | 1.1、1.2 | `AGENTS.md:16`；`AGENTS.md:82`；共享规范第 21、33 行 | 人工核对层级与 QEC Contract Rules | PASS | 共享规则通用，项目规则补充 Source of Truth 与领域约束 |

## Command Results

| 命令 | 退出码 | 实际结果 | 证据路径 |
| --- | ---: | --- | --- |
| `rg -n "Work Scope by User Intent|Requirement-to-Evidence Chain|Change Delivery Gates|Archive Blockers" AGENTS.md` | 0 | 4 个项目工作流章节全部存在 | `AGENTS.md` |
| `rg -n "用户意图|OpenSpec 项目|不得自动初始化|无需.*apply" /home/zephy/.codex/AGENTS.md` | 0 | 通用授权、条件式 OpenSpec 和无需重复 apply 均存在 | `/home/zephy/.codex/AGENTS.md` |
| `rg -n "Text Formatting|LaTeX|Registry key" /home/zephy/.codex/AGENTS.md` | 0 | 原有文本格式规范保持 | `/home/zephy/.codex/AGENTS.md` |
| `git diff --check` | 0 | 无空白或 patch 格式错误 | 工作区 diff |
| `openspec validate standardize-agent-workflow --strict --no-interactive` | 0 | `Change 'standardize-agent-workflow' is valid` | 本 change |
| `openspec validate --all --strict --no-interactive` | 0 | 11 passed，0 failed；1 条既有长 requirement INFO | 全部 specs 与 active changes |
| OpenSpec verify：按 Completeness、Correctness、Coherence 人工核对 requirement-to-evidence matrix | 0 | 0 CRITICAL、1 WARNING、0 SUGGESTION | 本文件 Verification Summary 与 Matrix |

本 change 只修改规范与验证模板，没有 Python 行为变化，因此未运行 Python 单元或集成测试。

## Evidence Integrity

| 产物 | 身份或 SHA-256 | 来源 |
| --- | --- | --- |
| `AGENTS.md` | `cfbe1aa3e6be0f5b44c74e3850833d9d81f5df99198a747387b5f26ab12bea55` | 项目级 Agent 规范 |
| `/home/zephy/.codex/AGENTS.md` | `e6a060c105c69229425230701cbfaa13e3ad5f91a2feef51d1e7848f745db1ff` | Zephy 级共享 Agent 规范 |
| `openspec/templates/verification.md` | `15211f0a0ea7cce768ce2a2d357029a2027255065f724280e6f58f1617ea2124` | verification evidence 模板 |
| change spec delta | `273829af00a451c3c5ae8bac6a8bce73d9e96442c424e6d38eb7f1c16a01f189` | `openspec/changes/standardize-agent-workflow/specs/governance/spec-management/spec.md` |

## Verification Findings

### CRITICAL

- 无。

### WARNING

- `/home/zephy/.codex/AGENTS.md` 位于项目 Git 仓库外，不能随仓库提交、归档或由 Git 回滚；本 change 以绝对路径文本检查和 SHA-256 固定其验证身份。项目内 `AGENTS.md`、主规格和 verification evidence 仍由 Git 管理。

### SUGGESTION

- 无。

## Limitations and Residual Risk

- OpenSpec 当前全局 profile 是 `core`，没有生成可直接调用的 `verify` workflow；本次按相同的 Completeness、Correctness、Coherence 维度完成了人工 verify。规范已经要求未来 change 执行 verify，但本 change 不扩展为全局 OpenSpec profile 配置修改。
- 共享规范未来若被独立修改，需要重新核对其与项目规范的语义一致性；项目内证据不会自动监控仓库外文件漂移。

## Verdict

- 结论：PASS WITH WARNINGS
- 允许归档：YES
- 原因：4 项任务和 9 个场景均有通过证据，严格 validation 与全仓 validation 通过，0 个 CRITICAL 问题；唯一 warning 是仓库外共享规范不受本仓库 Git 管理，已由摘要与限制明确记录，不影响当前规范生效。
