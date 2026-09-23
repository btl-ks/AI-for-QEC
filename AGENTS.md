# AI-for-QEC Agent Rules

## Source of Truth

按以下顺序读取项目上下文：

1. `openspec/specs/`
2. `openspec/changes/` 中相关 active change
3. `openspec/capabilities.yaml`
4. `README.md`
5. `docs/AI_for_QEC_Codex_Architecture_Context.md`
6. `docs/` 中的其他架构背景文档

OpenSpec 与架构背景冲突时，以 OpenSpec 为准。不得根据目标目录或接口骨架宣称功能已实现。

## OpenSpec Workflow

行为变更必须使用有界 OpenSpec change，并保持完整交付链：

```text
proposal → specs → design → tasks → implementation → tests → verification → archive
```

开始修改前读取相关主规格与 active changes。没有覆盖当前工作的 change 时，先创建或更新 proposal、spec delta、design 和 tasks；实现必须映射到命名 scenario、唯一 `<change-name>/<task-id>` 和可检查证据。

### Work Scope by User Intent

用户明确使用的动作动词是授权边界：

- **记录 / 文档化 / 更新说明**：把已确认的信息写入需求、决策或项目文档，不扩展为设计或实现。
- **分析 / 审查 / 调查**：只读检查当前状态并报告，不修改项目文件；用户同时要求记录或修复时，才执行对应写入。
- **设计 / 计划 / 提案 / 规格化**：创建或更新有界 change 的 proposal、specs、design、tasks 和追踪关系，并运行严格校验；不得修改实现代码。
- **实现 / 开发 / 修复 / 重构 / 完成**：授权所请求行为从规划到证据的完整生命周期。没有合适 change 时先完成并严格校验规划材料，然后直接继续实现、自动化测试、OpenSpec verify 和 verification evidence；不得要求用户再次发送 `apply`。
- **继续 / 恢复**：从已授权且名称明确的 change 的首个未完成任务继续，不重复请求授权。

请求包含多个动作动词时，执行其显式范围的并集。例如，“设计并实现”授权完整生命周期，“分析并记录”授权分析与文档写入，但不授权代码修改。应同时依据动词和请求对象判断边界；“修复文档”不授权无关代码变更。

纯规划请求即使已经产生可实施材料，也必须停在规划边界。实现授权跨规划步骤、上下文压缩和后续继续回合保持有效，直至 change 完成、暂停、取消或发生实质性改 scope。

### Requirement-to-Evidence Chain

每个可实施 change 必须保持以下追踪链：

1. **Requirement**：change specs 使用 SHALL/MUST 表达可观察行为，并定义命名验收场景。
2. **Functional design**：`design.md` 把需求映射到组件边界、API、数据与状态模型、失败语义、兼容或迁移决策以及成熟库/adapter 选择。
3. **Implementation task**：`tasks.md` 把设计拆成有界任务；每项任务引用对应 requirement/scenario、design 章节和具体证据。
4. **Test evidence**：每项任务声明自动化测试、验证命令或可检查产物；证据通过后才能勾选完成。

Umbrella change 可以建立跨项目需求基线，但实现前必须拆成可独立验证的有界 child changes。每个 child change 必须拥有自己的 spec delta、详细设计、实施任务和测试证据。

### Change Delivery Gates

每个可实施 child change 依次通过以下门禁：

1. 在 `proposal.md` 冻结有界 scope，并在 change specs 中写入可观察 requirements 与命名 scenarios。
2. 完成 `design.md`、`tasks.md` 及 Requirement-to-Evidence 追踪。
3. 实现前运行严格 OpenSpec validation，解决全部结构和语义错误。
4. 仅在用户意图授权实现时开始编码；原请求已经包含“实现、开发、修复、重构、完成”等同义动作时，严格校验后自动继续，无需第二次 `apply`。
5. 运行任务声明的测试和验证命令；证据通过后才勾选任务。
6. 运行 OpenSpec verify，检查 completeness、correctness、coherence；归档前解决全部 CRITICAL 问题。
7. 从 `openspec/templates/verification.md` 创建 change 的 `verification.md`，记录被测提交或文件摘要、环境、requirement/scenario/task/test 矩阵、精确命令与结果、证据路径与哈希、限制和最终 verdict。
8. 同步已接受的 delta specs，仅在证据支持时更新 `openspec/capabilities.yaml`，然后归档。

严格规划校验与实现验证是两个独立门禁：validation 只证明 proposal、specs、design 和 tasks 结构与语义有效；verification 才证明实现和测试满足这些材料。不得用前者替代后者。

常用命令：

```bash
openspec new change <kebab-case-name>
openspec status --change <name>
openspec instructions <artifact> --change <name>
openspec instructions apply --change <name>
openspec validate <name> --strict --no-interactive
openspec validate --all --strict --no-interactive
openspec archive <name> --yes
```

### Archive Blockers

存在以下任一情况时不得归档：任何 task 未勾选、任何必需 scenario 缺少通过证据、`verification.md` 缺失、verify report 仍有 CRITICAL 问题，或当前状态文档与实现不一致。Warnings 和已接受限制必须写入 `verification.md`，并说明其对归档 verdict 的影响。

## Contract Rules

- 公共 Notebook API 必须从 `ai_qec/notebook_api.py` 导出。
- 核心 contract 保持 vendor-neutral；具体工具只能位于 adapter/implementation 层。
- 不得创建空数据、零样本或占位输出伪造成功。
- 不得静默 fallback backend 或静默近似 NoiseSpec。
- DatasetArtifact 一旦提交即不可变；消费前必须校验完整性。
- Failed/Interrupted Attempt 不得原地恢复为成功；恢复必须创建新 Attempt。
- ModelCheckpoint 与 TrainingRecoveryCheckpoint 必须分离。
- Scientific evaluation 与 performance evaluation 必须分离。
- 没有实现、测试与 verification evidence 时，capability 必须保持 `planned`。
- 所有由配置选择的实现必须通过 `REGISTRIES_BY_PATH` 对应 Registry 构造，不得在业务层散落第三方名称分支。
- 字符串 `unresolved` 是唯一未决占位值；任何构造前必须 fail fast，只有当前阶段不消费的精确字段路径可以显式 allow。
- Protocol 或 technology catalog 条目不得注册为可执行工厂；Registry 只登记真实可用实现。

## Text Formatting

- 对话窗口中的回复不强制使用 LaTeX。简单变量、参数值、数量、单位、范围和倍数优先使用可直接阅读的普通文本，例如 `L=6`、`p=0.10`、`10,000 shots`、`7–8.5 s`、`5.3×`，避免在表格中显示未渲染的 `$...$` 或 `\mathrm{...}`。
- 写入项目的 Markdown 与 Jupyter Markdown 文档时，真正需要数学排版的公式和表达式仍使用 LaTeX：行内公式使用 `$...$`，独立公式使用 `$$...$$`，例如 `$Z_L^{(1)}$`。简单参数和带单位的数据可以使用普通文本，以清晰、稳定渲染为准。
- 程序标识符、配置字段、Registry key、命令和文件路径仍使用代码格式，不得误写为数学公式。

## Long-Running Execution

- 预计超过几分钟的运行（执行论文 Notebook、训练网格、批量评估）必须用 `setsid nohup` 启动，与 agent 会话脱离，stdout/stderr 写入日志文件；不得作为 agent 会话的后台任务运行。编辑器窗口重载或会话结束会杀掉会话的整个进程树。
- 启动后记录 PID 与日志路径，另起等待命令监视该 PID 结束；汇报结果前先读日志，并检查相关 Attempt 与 Stage 状态。
- 运行被中断后，重新执行同一入口以创建恢复 Attempt；不得手工修改 `runs/` 中的 Attempt 或 Stage 记录。

```bash
# setsid 可能 fork，$! 不可靠；由进程自己写 PID，exec 保持同一 PID
setsid nohup bash -c 'echo $$ > <pid-file>; exec <python> <entry>' > <log> 2>&1 < /dev/null &
```

## Scope Discipline

优先交付 Dataset reuse、Attempt recovery、Scientific Accuracy Gate 与 Training/Execution 分离。除非 active change 明确授权，不要提前加入 Kubernetes、Ray、DeepSpeed、FSDP、TensorRT server、FPGA 板级实现、ASIC 物理设计或实时 QPU feedback。
