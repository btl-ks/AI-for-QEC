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

## Scope Discipline

优先交付 Dataset reuse、Attempt recovery、Scientific Accuracy Gate 与 Training/Execution 分离。除非 active change 明确授权，不要提前加入 Kubernetes、Ray、DeepSpeed、FSDP、TensorRT server、FPGA 板级实现、ASIC 物理设计或实时 QPU feedback。
