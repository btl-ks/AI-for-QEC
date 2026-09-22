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

## Required Workflow

行为变更必须使用有界 OpenSpec change：

```text
proposal → specs → design → tasks → implementation → tests → verification → archive
```

常用命令：

```bash
openspec new change <kebab-case-name>
openspec status --change <name>
openspec instructions <artifact> --change <name>
openspec validate <name> --strict --no-interactive
openspec validate --all --strict --no-interactive
```

归档前必须满足：任务全部完成、相关测试通过、`verification.md` 有真实证据、capability 状态与证据一致。

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

- 所有面向读者的文本中的数学公式、数学变量和数学表达式必须使用 LaTeX 格式；Markdown 与 Jupyter Markdown 中的行内公式使用 `$...$`，独立公式使用 `$$...$$`，例如 `$Z_L^{(1)}$`。
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
