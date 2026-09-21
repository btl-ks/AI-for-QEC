# Design

## Context

当前 `ai_qec` 只提供 contracts，且特意不发布假的 `create_experiment()` 执行函数。用户给出的目标代码包含高层 `start_or_recover()`、`resolve_dataset()` 和评估方法，因此需要在不改变“无实现”事实的情况下保存该工作流。

## Goals / Non-Goals

**Goals:**

- 提供一份真实、可解析、可 clean `Run All` 的 Notebook 源文件。
- 用公共 Protocol 固定用户期望的高层编排形状。
- 维持科学评估、Accuracy Gate 和性能评估的清晰边界。

**Non-Goals:**

- 不让 Notebook 产生数据、训练模型或执行基线。
- 不增加 Stim、PyTorch、CUDA、Jupyter 或绘图库运行时依赖。
- 不声称完成论文复现，不创建结果图和占位 Artifact。

## Decisions

### 1. 使用显式 runtime 注入

新增 `NotebookPlatform`、`NotebookExperiment` 和 `NotebookRun` Protocol。Notebook 定义 `run_paper(runtime)`，通过 `runtime.create_experiment(CONFIG)` 启动目标流程，但默认不构造或调用 runtime。

与直接发布 `qec.create_experiment()` 占位函数相比，此设计不会在缺少实现时返回 `None`、抛出占位异常或让用户误以为功能可用。

### 2. 显式调用 Accuracy Gate

目标示例中的 `scientific_result.accuracy_gate_passed` 调整为独立的 `run.check_accuracy_gate(scientific_result)`。这保持 ScientificEvaluationResult 与 ScientificAcceptanceResult 分离，并且性能评估仅在 `GateDecision.PASS` 时调用。

### 3. 性能结果暂时使用 ArtifactRef

当前没有 PerformanceEvaluationSpec 的正式 capability。本次高层接口将性能输出表示为通用 `ArtifactRef`，不提前定义 latency、throughput 或硬件报告结构。未来性能 change 可替换为专门的 vendor-neutral contract。

### 4. Notebook 保持纯 JSON 和标准库可验证

直接保存 nbformat v4 JSON；测试使用标准库解析 JSON、编译每个 code cell，并在共享 namespace 中执行 cells。Notebook 不要求安装 Jupyter 才能完成 CI 验证。

## Risks / Trade-offs

- [接口与未来 runtime 不一致] → 通过独立 OpenSpec change 演进 Protocol，并保持 Notebook 只依赖 facade。
- [用户误认为 Notebook 已能研究复现] → README、Notebook 首屏和 capability registry 明确标记 `contracts-only`。
- [ArtifactRef 不能表达完整性能报告] → 本次只固定调用顺序，性能 contract 留给后续 capability。

## Migration Plan

1. 新增 paper Protocol 并从公共 facade 导出。
2. 新增 Notebook 与目录说明。
3. 验证 JSON、code cells、clean execution、公共导入和全量 OpenSpec。
4. 归档 change 并把 `research/paper-reproduction` 登记为 `planned/contracts-only`。
