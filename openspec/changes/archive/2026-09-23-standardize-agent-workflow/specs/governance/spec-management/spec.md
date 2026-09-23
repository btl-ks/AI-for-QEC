# Spec Delta

## ADDED Requirements

### Requirement: Agent 工作范围由用户意图约束
Agent SHALL 将用户明确使用的动作动词解释为授权边界：记录类请求只写入既有信息，分析类请求只读检查，设计类请求只生成并校验规划材料，实现类请求授权从规划到证据的完整生命周期，继续类请求从已授权 change 的首个未完成任务继续。多个动作动词同时出现时，Agent SHALL 执行其显式范围的并集；设计类请求 MUST NOT 因规划材料已经可执行而自动进入实现，实现类请求 MUST NOT 要求用户在规划完成后重复发送 `apply`。

#### Scenario: 用户只要求设计
- **WHEN** 用户要求设计、计划、提案或规格化一个行为变更，且没有同时授权实现
- **THEN** Agent SHALL 创建或更新有界 change 的 proposal、specs、design、tasks 与追踪关系并完成严格校验，但 MUST NOT 修改实现代码

#### Scenario: 用户要求完成实现
- **WHEN** 用户要求实现、开发、修复、重构或完成一个行为变更
- **THEN** Agent SHALL 将该请求视为完整生命周期授权，在规划材料严格校验通过后直接继续实现、测试、验证与证据记录，无需再次请求 `apply`

#### Scenario: 用户要求分析并记录
- **WHEN** 用户同时要求分析现状并把结论记录到项目文档
- **THEN** Agent SHALL 执行只读分析并写入所请求的文档，但 MUST NOT 扩展为未授权的设计或代码变更

### Requirement: 可实施 change 保持需求到证据追踪
每个可实施 OpenSpec change SHALL 保持 Requirement → Functional design → Implementation task → Test evidence 的完整追踪链；每个 task MUST 引用对应 requirement/scenario、design 边界与可执行或可检查的证据，且只有证据通过后才能标记完成。

#### Scenario: 编写实施任务
- **WHEN** Agent 为有界 change 编写 `tasks.md`
- **THEN** 每项任务 SHALL 指明对应需求或场景、设计章节以及验证命令或可检查产物

#### Scenario: 证据尚未通过
- **WHEN** 某项任务的声明测试、验证命令或检查产物尚未成功
- **THEN** 该任务 MUST 保持未完成状态，相关 capability 状态 MUST NOT 因文件或接口已经存在而升级

### Requirement: OpenSpec 交付门禁区分规划校验与实现验证
可实施 change SHALL 依次完成有界范围、规划材料、严格规划校验、实现、自动化测试、完整性/正确性/一致性验证、`verification.md`、主规格同步与归档。严格规划校验 SHALL 证明 proposal、specs、design 与 tasks 的结构和语义有效；实现验证 SHALL 独立证明代码与测试满足这些材料。任一任务未完成、必需场景缺少通过证据、`verification.md` 缺失、存在未解决的 CRITICAL 问题或当前状态文档与实现不一致时，change MUST NOT 归档。

#### Scenario: 规划严格校验通过
- **WHEN** proposal、specs、design 与 tasks 通过严格 OpenSpec validation
- **THEN** Agent SHALL 仅把规划门禁标记为通过，不得据此宣称实现或 capability 已经可用

#### Scenario: 归档前存在阻断项
- **WHEN** change 仍有未完成任务、无证据场景、缺失的 `verification.md`、未解决的 CRITICAL 问题或状态与实现不一致
- **THEN** Agent MUST NOT 归档该 change，并 SHALL 明确报告阻断项

### Requirement: 共享与项目 Agent 规范分层
Zephy 级共享 Agent 规范 SHALL 定义通用用户意图边界以及 OpenSpec 项目的通用交付门禁，但 MUST NOT 强迫未采用 OpenSpec 的项目初始化或使用 OpenSpec。项目 `AGENTS.md` SHALL 在共享规则之上声明项目特有的 Source of Truth、contract、验证和运行约束；发生冲突时，Agent SHALL 遵循适用的更高优先级指令与更具体的项目约束。

#### Scenario: 非 OpenSpec 项目收到实现请求
- **WHEN** Agent 在没有声明采用 OpenSpec 的项目中收到实现请求
- **THEN** 共享规范 MUST NOT 要求 Agent 自动运行 `openspec init` 或创建 `openspec/` 目录

#### Scenario: QEC 项目收到行为变更请求
- **WHEN** Agent 在本项目处理行为变更
- **THEN** Agent SHALL 同时遵守共享的用户意图边界与本项目的 OpenSpec、capability evidence、QEC contract 和长时运行约束
