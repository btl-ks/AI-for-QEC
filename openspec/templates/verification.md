# Verification Evidence: <change-name>

> 实现和自动化测试完成后，将本模板复制为 `openspec/changes/<change-name>/verification.md`。所有占位符必须替换为实际观测证据；计划结果或模拟结果不能用于勾选任务。

## Verification Identity

- Change：<change-name>
- Schema：<schema-name>
- 被测提交：<full-commit-sha>
- 分支：<branch-name>
- 验证时间：<ISO-8601 timestamp>
- 验证者：<person-or-agent>

## Environment

- 操作系统与运行时：<values>
- Python 与依赖锁身份：<values>
- 加速器、驱动与后端：<values-or-not-applicable>
- 数据集身份与 manifest 哈希：<values-or-not-applicable>
- 配置快照与哈希：<values-or-not-applicable>

## Scope

- 纳入的 requirements 与 scenarios：<list>
- 排除项及原因：<list-and-reason>
- OpenSpec task identities：<change-name/task-id-list>

## Verification Summary

| 维度 | 实际结果 | 状态 |
| --- | --- | --- |
| Completeness | <已完成任务数/总任务数及 requirements> | PASS/WARN/FAIL |
| Correctness | <已覆盖 requirements/scenarios 数量> | PASS/WARN/FAIL |
| Coherence | <设计一致性与项目模式检查结果> | PASS/WARN/FAIL |

- CRITICAL 问题：<count>
- WARNING 问题：<count>
- SUGGESTION 问题：<count>

## Requirement-to-Evidence Matrix

| Requirement | Scenario | Design 章节 | Task | 实现位置 | 自动化测试或命令 | 结果 | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <stable requirement name> | <stable scenario name> | <design heading> | <task id> | <path:line> | <exact test or command> | PASS/FAIL | <report, log, metric, or artifact path> |

每个必需 scenario 至少填写一行。只有该 requirement 的所有必需 scenarios 都有通过证据时，才视为覆盖。

## Command Results

| 命令 | 退出码 | 实际结果 | 证据路径 |
| --- | ---: | --- | --- |
| <exact command> | <code> | <concise observed result> | <path-or-not-applicable> |

必须包括 OpenSpec 严格校验、聚焦自动化测试、要求的冒烟或回归测试，以及 OpenSpec verify 结果。

## Evidence Integrity

| 产物 | 身份或 SHA-256 | 来源 |
| --- | --- | --- |
| <path> | <hash-or-stable-id> | <producing command, run id, or manifest> |

不要提交生成的数据集、runs、checkpoints 或论文发布产物。它们保留在指定存储中，此处记录稳定身份与哈希。

## Verification Findings

### CRITICAL

- <无，或带文件与行号的可执行问题>

### WARNING

- <无，或带文件与行号的可执行问题>

### SUGGESTION

- <无，或带文件与行号的可执行问题>

## Limitations and Residual Risk

- <已知限制、跳过的检查及其影响>

## Verdict

- 结论：PASS / PASS WITH WARNINGS / FAIL
- 允许归档：YES / NO
- 原因：<基于证据的结论>
