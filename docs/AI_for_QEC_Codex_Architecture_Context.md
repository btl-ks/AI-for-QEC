# AI-for-QEC 项目架构与 Codex 实施上下文

> 目标读者：Codex / Coding Agent / 项目维护者  
> 项目：`btl-ks/AI-for-QEC`  
> 文档目的：固化当前已经达成的架构决策，避免依赖聊天上下文；为后续 OpenSpec 设计、实现、测试和迁移提供统一上下文。  
> **重要：本文件是架构上下文，不替代 OpenSpec。`openspec/` 仍然是正式需求、变更、验收和证据的唯一真相源。**

---

## 1. Codex 必须先遵守的项目规则

任何设计或实现前，先阅读：

1. `AGENTS.md`
2. `openspec/capabilities.yaml`
3. 相关 `openspec/specs/`
4. 相关 active `openspec/changes/`
5. `README.md`
6. `configs/experiment.smoke.yaml`
7. `scripts/run_experiment.py`
8. `ai_qec/`

### 1.1 Source of Truth 层级

```text
OpenSpec requirements / active changes
        ↓
Architecture context / design intent
        ↓
Python implementation
        ↓
Tests / verification evidence
        ↓
Capability registry state
```

规则：

- 不允许因为本文件描述了“目标架构”就直接宣称功能已实现。
- 行为变化必须遵循 `AGENTS.md` 中的 OpenSpec workflow。
- `openspec/capabilities.yaml` 只有在实现、测试和 verification evidence 一致时才能升级状态。
- 所有新 public API 必须通过 `ai_qec/notebook_api.py`（导入名 `ai_qec.notebook_api`）暴露。
- 不能用空文件、占位数组、零样本数据伪造成功 pipeline。
- 正式科学结论必须由论文或项目内可审计实验结果支持。

---

# 2. 项目最高层研究逻辑

项目不是一个泛化 MLOps 平台，也不是“另一个 PyTorch / CUDA-Q”。

平台自己拥有的是：

```text
QEC semantics
Dataset semantics
Experiment semantics
Scientific evaluation contracts
Provenance
Verification
```

底层执行尽量复用成熟工具：

```text
QEC simulation      → Stim / Sinter / CUDA-Q QEC / cuStabilizer
AI training         → PyTorch
Multi-GPU           → torchrun / DDP
GPU optimization    → torch.compile / ONNX / TensorRT
Profiling           → torch.profiler / Nsight / DCGM
Classical baseline  → MWPM / PyMatching / Tesseract
FPGA / ASIC         → HLS / RTL / synthesis toolchains
```

总体原则：

> **Own the semantics and contracts; reuse execution infrastructure.**

技术方向：

> **NVIDIA-first, vendor-neutral core.**

即优先利用 NVIDIA 生态，但 `QECSpec`、`NoiseSpec`、`DatasetArtifact`、`DecodeRequest`、`DecodeResult`、`PerformanceReport` 等核心 contract 不绑定 NVIDIA。

---

# 3. 研究 Golden Path

```text
1. Configure
      ↓
2. Dataset Resolve / Reuse
      ↓
3. AI Training
      ↓
4. Scientific Evaluation
      ↓
5. Traditional Baseline Comparison
      ↓
6. Accuracy Gate
   ┌───────────────┐
   │               │
 FAIL             PASS
   │               │
   ↓               ↓
Model/Data      Freeze Golden Model
Research            ↓
              7. Performance Research
                 ├── GPU
                 ├── FPGA
                 └── ASIC
                       ↓
              8. Accuracy Revalidation
                       ↓
              9. Trade-off Analysis
                       ↓
             10. Visualization / Report
```

核心研究原则：

> **先解决 AI decoder 是否足够正确，再解决是否足够快。**

```text
Phase I  = Scientific Accuracy
Phase II = Performance
```

正式性能研究只有在科学准确率达到预定义传统算法基线后才进入。

---

# 4. Scientific Accuracy 与 Accuracy Gate

“准确率”不是只指普通 classification accuracy。QEC decoder 的核心科学指标优先是：

```text
Logical Error Rate (LER)
```

并结合：

```text
failure count
shots
confidence interval
timeout / convergence
calibration
failure distribution
generalization
```

所有 AI / classical decoder 对比应遵守：

```text
same QEC problem
same noise
same test DatasetArtifact
same shot set
same observable truth
same stopping semantics
same LER definition
```

建议正式抽象：

```text
AccuracyGate
|
+-- primary_metric
|   └── logical_error_rate
|
+-- baseline
|   └── MWPM / PyMatching / declared classical reference
|
+-- protocol
|   ├── same dataset
|   ├── same code
|   ├── same noise points
|   └── same shots
|
+-- statistical_rule
|   ├── confidence interval
|   └── preregistered tolerance
|
└── result
    ├── PASS
    └── FAIL
```

### Gate A

```text
trained model
    ↓
Scientific Evaluation
    ↓
Gate A PASS
    ↓
Golden Model
```

### Gate B

模型经过量化、剪枝、fixed point、FPGA/ASIC mapping 等优化后必须重新验证：

```text
Golden Model
    ↓
Optimization Candidate
    ↓
Scientific Re-evaluation
    ↓
Gate B
```

性能更快但 LER 不满足约束的 candidate 不应被标记为正式可接受部署结果。

---

# 5. 当前仓库实际状态

当前仓库主要结构：

```text
AI-for-QEC/
├── ai_qec/
├── configs/
├── openspec/
├── paper/
├── scripts/
├── tests/
├── README.md
├── AGENTS.md
├── pyproject.toml
└── uv.lock
```

当前 `ai_qec/` 已有：

```text
ai_qec/
├── adaptation/
├── benchmarks/
├── data/
├── deployment/
├── design/
├── models/
├── qec/
├── runtime/
├── training/
├── utils/
└── notebook_api.py
```

当前已有的有价值基础：

```text
OpenSpec governance
Run / Stage recording
artifact hash verification
named random streams
Torlai–Melko notebook orchestration
RBM training
RBM Gibbs decoder
Exact MWPM reference
PyMatching fallback
same-test-split comparison
LER / Wilson CI
source checkpoint reuse
unit tests
smoke tests
paper reproduction protocol
```

不要推翻这些基础。

---

# 6. 当前 capability 状态

当前 `openspec/capabilities.yaml`：

```text
validated
├── governance/spec-management
└── research/experiment-lifecycle

available
└── research/paper-reproduction

planned
├── qec/dataset-pipeline
├── qec/decoder-workbench
├── research/data-efficient-training
├── research/automated-validation
└── deployment/performance-runtime
```

Requirement 存在不代表实现完成；新功能必须通过 change → code → tests → verification → capability evidence 流程交付。

---

# 7. 当前设计与目标设计的四个主要冲突

## 7.1 Dataset reuse

当前正式 requirement 明确要求跨 run 不复用 dataset，当前 `dataset_resolution.py` 也按 run-owned dataset 实现。

目标：

```text
DatasetSpec
    ↓
DatasetKey
    ↓
DatasetRegistry
   /        \
 HIT        MISS
  ↓          ↓
reuse      generate
   \        /
   DatasetArtifact
        ↓
DatasetInstance
        ↓
       Run
```

这必须先修改 OpenSpec，再改代码。

## 7.2 Crash recovery

当前 validated lifecycle 明确规定中断执行不得断点续跑、checkpoint 不携带恢复状态。

目标：

```text
Experiment 可以恢复
失败 Run/Attempt 不原地复活
恢复创建新的 Attempt
```

例如：

```text
Experiment E001
|
+-- Run R001
|   ├── dataset COMPLETE
|   ├── training FAILED
|   └── terminal = FAILED / INTERRUPTED
|
└-- Run R002
    ├── recovery_from = R001
    ├── reuse verified DatasetArtifact
    ├── restore verified training recovery state
    └── continue
```

必须区分：

```text
ModelCheckpoint
≠
TrainingRecoveryCheckpoint
≠
RecoveryState
≠
Run
```

## 7.3 Accuracy Gate 尚未是一等对象

需要补：

```text
ScientificEvaluationSpec
AccuracyGateSpec
ScientificAcceptanceResult
```

## 7.4 TrainingSpec 与 ExecutionSpec 需要分离

目标：

```text
TrainingSpec
├── optimizer
├── learning_rate
├── epochs
├── batch_size
├── scheduler
└── loss

ExecutionSpec
├── device
├── cpu resources
├── gpu_count
├── DDP
├── num_workers
├── AMP
└── torch.compile
```

更换 1 GPU → 4/8 GPU DDP 不应改变 dataset identity。

---

# 8. Dataset 三层模型

```text
DatasetSpec
DatasetArtifact
DatasetInstance
```

## DatasetSpec

描述“应该生成什么数据”。典型字段：

```text
QEC identity
Noise identity
generator
generator version
backend semantics
sample counts
seed
split policy
schema
preprocessing
```

## DatasetArtifact

物理、不可变、可验证、可跨 Run 复用的数据实体：

```text
DatasetArtifact DA-abc123
├── manifest
├── generation provenance
├── checksums
├── train shards
├── validation shards
└── test shards
```

要求：

```text
immutable
content-addressed / deterministic identity
checksum verified
provenance preserved
```

## DatasetInstance

Run 本地绑定：

```text
Run R001
    ↓
DatasetInstance DI-R001
    ↓
DatasetArtifact DA-abc123

Run R002
    ↓
DatasetInstance DI-R002
    ↓
DatasetArtifact DA-abc123
```

---

# 9. Dataset identity 边界

应该参与 DatasetKey：

```text
QECSpec
NoiseSpec
generator
generator version
sampling semantics/backend identity when scientifically relevant
shots / sample counts
seed
split policy
preprocessing affecting stored data
schema / data representation
```

通常不参与：

```text
ModelSpec
TrainingSpec
GPU count
DDP
AMP
num_workers
torch.compile
EvaluationSpec
VisualizationSpec
```

因此同一 DatasetArtifact 可以被 CNN / GNN / Transformer 和不同 GPU 并行配置复用。

---

# 10. Experiment / Run / Attempt / Stage

推荐语义：

```text
Experiment
|
├── Run / Attempt R001
│   ├── Stage: dataset
│   ├── Stage: train
│   ├── Stage: scientific_evaluation
│   ├── Stage: accuracy_gate
│   └── ...
|
└── Run / Attempt R002
    └── recovery_from = R001
```

Run/Attempt 终态应明确：

```text
COMPLETED
FAILED
INTERRUPTED
PARTIAL
```

Stage 至少记录：

```text
stage_id
status
started_at
finished_at
duration
inputs
outputs
artifact hashes
error
recovery metadata
```

---

# 11. Notebook recovery 原则

Notebook 恢复单位是 **Stage**，不是 Cell。

禁止：

```text
remember cell index
restore Python kernel memory
resume from cell N
```

目标行为：

```text
Kernel restart
    ↓
Run All
    ↓
load experiment identity
    ↓
inspect previous Attempt
    ↓
verify completed artifacts
    ↓
reuse safe completed stages
    ↓
create new Attempt for recovery
    ↓
resume from safe Stage / checkpoint
```

v0.1 推荐：

```text
stage-boundary experiment recovery
epoch-boundary training recovery
```

先不做 mid-batch 精确恢复。

---

# 12. Scientific reuse 与 Failure recovery

当前 notebook 的 `REUSE_CHECKPOINT_FROM_RUN` 属于 Scientific Reuse：主动复用一个成功 source run 的模型重新评估。

Failure recovery 属于 Execution Recovery：前一个 Attempt 异常中断，需要恢复同一个 Experiment。

两者不能共用模糊语义。

---

# 13. Model checkpoint 与 Recovery checkpoint

## ModelCheckpoint

用于：

```text
evaluation
inference
scientific reuse
deployment
```

可包含：

```text
model weights
model identity
input/schema identity
dataset/config provenance
selected metric
```

## TrainingRecoveryCheckpoint

用于从中断训练恢复到新的 recovery Attempt，可包含：

```text
model state
optimizer state
scheduler state
AMP scaler
epoch
step
RNG state
sampler/training cursor if needed
```

它不是普通 ModelArtifact。

---

# 14. Named random streams

需要稳定的具名流，例如：

```text
qec_sampling
dataset_split
model_init
training_shuffle
adaptive_sampling
```

要求：

```text
same master seed + same stream name → same sequence
different stream names → independent state
consuming one stream → MUST NOT alter another stream
```

当前项目已有 random stream 基础，应复用/迁移，不重写语义。

---

# 15. Notebook 职责边界

Notebook 负责：

```text
paper metadata
experiment parameters
stage ordering
small paper-specific glue
visualization
interpretation
```

Notebook 不应复制：

```text
dataset generator implementation
training loop
decoder core loop
artifact management
run state machine
baseline implementation
```

推荐入口：

```python
import ai_qec.notebook_api as qec
```

概念目标 API：

```python
config = qec.ExperimentConfig(...)

experiment = qec.create_experiment(config)
run = experiment.start_or_recover()

dataset = run.resolve_dataset()
model = run.train(dataset)

evaluation = run.evaluate(
    model=model,
    dataset=dataset,
    baselines=["mwpm"],
)

acceptance = run.check_accuracy_gate(evaluation)
run.visualize(evaluation)
run.finish()
```

具体 API 名称必须通过 OpenSpec design 决定。

---

# 16. 当前 Torlai–Melko Notebook 的目标生命周期

当前真实 notebook 近似：

```text
1. Experiment Configuration
2. Start Run
3. Generate Dataset
4. Train RBM
5. Neural Decode
6. MWPM Baseline
7. Visualization / Finish
```

目标：

```text
1. Configure
2. Start / Recover Experiment
3. Resolve Dataset
   ├── cache hit → verify + reuse
   └── cache miss → generate + register
4. Train
   ├── normal
   └── recover from prior failed attempt
5. Scientific Evaluation
6. Classical Baseline
7. Accuracy Gate
8. Visualization / Evidence
```

`clean kernel + Run All` 必须仍然成立。

---

# 17. QEC 数据 contract

长期统一数据输入：

```text
QECBatch
|
├── detector events / syndrome
├── observable truth / target
├── measurement history
├── optional physical error
├── detector coordinates
├── round mask
├── sample ids
├── dataset artifact id
└── provenance/context
```

```text
Stim --------\
CUDA-Q -------→ QECBatch → Trainer / Decoder
QPU ----------/
```

模型不应知道数据来自哪个 simulator。

---

# 18. 数据生成后端

```text
QECSpec + NoiseSpec
        ↓
Generation Adapter
   /       |       \
Stim    CUDA-Q    QPU/Qiskit
```

角色：

```text
Stim / Sinter
└── fast CPU QEC reference / sampling

CUDA-Q QEC / cuStabilizer
└── NVIDIA GPU-native generation path

Qiskit
└── future hardware adapter, not primary simulator
```

Noise adapter 禁止静默近似；无法精确表示时必须 reject 或 explicit approximation。

---

# 19. CPU / GPU 数据流水线

CPU 路径：

```text
Stim workers
    ↓
IterableDataset
    ↓
PyTorch DataLoader
    ↓
prefetch
    ↓
pinned memory
    ↓
non-blocking H2D
    ↓
GPU training
```

优先使用：

```text
num_workers
prefetch_factor
persistent_workers
pin_memory
non_blocking=True
```

只有 profiler 证明 DataLoader 是瓶颈后再考虑 DALI。

GPU-native 路径：

```text
CUDA-Q QEC / cuStabilizer
    ↓
GPU-native QEC batch
    ↓
PyTorch CUDA
```

不要强行 GPU → CPU → DataLoader → GPU。

---

# 20. Multi-GPU training

单机多 GPU 首选：

```text
torchrun
DistributedDataParallel (DDP)
```

模型能放进单 GPU 时优先 DDP。未来确有需要再考虑 FSDP2 / Tensor Parallel / Pipeline Parallel / DeepSpeed。

---

# 21. Performance Research

正式 performance phase 应在 Gate A PASS 后进入。

GPU 候选：

```text
PyTorch CPU
PyTorch CUDA
torch.compile
ONNX Runtime
TensorRT
```

性能报告至少区分：

```text
model latency
decoder latency
end-to-end latency
```

并记录：

```text
hardware
software versions
input shape
batch
warmup
synchronization boundaries
repetitions
p50
p95
p99
throughput
memory
```

### FPGA

```text
Golden Model
    ↓
quantization
    ↓
fixed point
    ↓
HLS / FINN / hls4ml / vendor tools
    ↓
HLS simulation
    ↓
RTL simulation
    ↓
synthesis
    ↓
board measurement
```

### ASIC

```text
Golden Model
    ↓
quantized / fixed-point spec
    ↓
HLS / RTL
    ↓
simulation
    ↓
synthesis
    ↓
timing / area / power
```

硬件结果必须区分 software projection / simulation / synthesis estimate / real-device measurement。

---

# 22. Accuracy–Performance Trade-off

最终优化不是只找最快或最准，而是联合研究：

```text
LER
latency
throughput
memory
power
FPGA resources
ASIC area
```

抽象：

```text
OptimizationCandidate
TradeoffStudy
Pareto Frontier
```

---

# 23. 目标目录架构

标记：

```text
[E] = 当前已有，原则上保留
[R] = 当前已有，但建议重构/迁移
[N] = 建议新增
[F] = Future，不急于实现
[G] = Runtime generated / Git ignored
```

```text
AI-for-QEC/
│
├── README.md                                      [E]
├── AGENTS.md                                      [E]
├── pyproject.toml                                 [E]
├── uv.lock                                        [E]
├── .gitignore                                     [E]
├── .gitattributes                                 [E]
│
├── openspec/                                      [E]
│   ├── config.yaml
│   ├── capabilities.yaml
│   ├── capabilities.schema.json
│   ├── research-catalog.yaml
│   ├── research-catalog.schema.json
│   ├── technology-catalog.yaml
│   ├── technology-catalog.schema.json
│   ├── paper-protocol.schema.json
│   ├── legacy-docs.yaml
│   ├── specs/
│   │   ├── governance/spec-management/spec.md
│   │   ├── research/experiment-lifecycle/spec.md
│   │   ├── research/paper-reproduction/spec.md
│   │   ├── research/data-efficient-training/spec.md
│   │   ├── research/automated-validation/spec.md
│   │   ├── qec/dataset-pipeline/spec.md
│   │   ├── qec/decoder-workbench/spec.md
│   │   └── deployment/performance-runtime/spec.md
│   ├── changes/
│   │   ├── <active-change>/
│   │   │   ├── proposal.md
│   │   │   ├── design.md
│   │   │   ├── tasks.md
│   │   │   ├── verification.md
│   │   │   └── specs/
│   │   └── archive/
│   ├── papers/
│   │   ├── torlai-melko-2017.yaml
│   │   └── <future-paper>.yaml
│   └── templates/verification.md
│
├── ai_qec/                                        [E]
│   ├── notebook_api.py                            [E]
│   │
│   ├── experiment/                                [N]
│   │   ├── spec.py
│   │   ├── plan.py
│   │   ├── experiment.py
│   │   ├── run.py
│   │   ├── stage.py
│   │   ├── state.py
│   │   ├── recovery.py
│   │   ├── random_streams.py
│   │   ├── artifact.py
│   │   ├── provenance.py
│   │   └── integrity.py
│   │
│   ├── qec/                                       [E]
│   │   ├── codes/
│   │   │   ├── base.py
│   │   │   ├── registry.py
│   │   │   ├── stabilizer.py
│   │   │   ├── repetition_code.py
│   │   │   ├── surface_code.py
│   │   │   └── toric_code.py
│   │   ├── noise/
│   │   │   ├── base.py
│   │   │   ├── registry.py
│   │   │   ├── pauli.py
│   │   │   ├── phase_flip.py
│   │   │   ├── measurement.py
│   │   │   ├── correlated.py
│   │   │   ├── crosstalk.py
│   │   │   └── leakage.py
│   │   ├── circuits/
│   │   │   ├── base.py
│   │   │   ├── registry.py
│   │   │   ├── code_capacity.py
│   │   │   └── memory.py
│   │   ├── detectors/
│   │   │   ├── syndrome.py
│   │   │   ├── detector_graph.py
│   │   │   └── dem.py
│   │   └── backends/                              [R/N]
│   │       ├── protocol.py
│   │       ├── stim.py
│   │       ├── cudaq.py
│   │       ├── qiskit.py                          [F]
│   │       ├── toric_backend.py
│   │       └── custom_backend.py
│   │
│   ├── data/                                      [E/R]
│   │   ├── schema/
│   │   │   ├── sample.py
│   │   │   ├── batch.py                           [N]
│   │   │   ├── context.py
│   │   │   └── manifest.py
│   │   ├── datasets/
│   │   │   ├── spec.py                            [N]
│   │   │   ├── identity.py                        [N]
│   │   │   ├── artifact.py                        [N]
│   │   │   ├── instance.py                        [N]
│   │   │   ├── registry.py                        [R]
│   │   │   ├── resolution.py                      [R]
│   │   │   ├── storage.py                         [N]
│   │   │   ├── validation.py                      [N]
│   │   │   ├── qec_dataset.py
│   │   │   └── toric_dataset.py
│   │   ├── generators/
│   │   │   ├── protocol.py                        [N]
│   │   │   ├── qec_generator.py
│   │   │   ├── paired_generator.py
│   │   │   ├── toric_generator.py
│   │   │   ├── stim_generator.py                  [N]
│   │   │   └── cudaq_generator.py                 [N]
│   │   ├── splits/
│   │   │   ├── split.py
│   │   │   └── audit.py                           [N]
│   │   ├── preprocessing/
│   │   ├── sampling/
│   │   └── loaders/                               [N]
│   │       ├── pytorch.py
│   │       ├── streaming.py
│   │       ├── gpu_native.py
│   │       └── dali.py                            [F]
│   │
│   ├── models/                                    [E]
│   │   ├── registry.py
│   │   ├── decoders/
│   │   │   ├── protocol.py
│   │   │   ├── classical/
│   │   │   │   ├── mwpm.py
│   │   │   │   ├── pymatching_adapter.py
│   │   │   │   └── tesseract_adapter.py          [F]
│   │   │   ├── generative/
│   │   │   │   ├── rbm.py
│   │   │   │   └── rbm_decoder.py
│   │   │   ├── neural/                            [N]
│   │   │   │   ├── cnn/{model.py,decoder.py}
│   │   │   │   ├── gnn/{model.py,decoder.py}
│   │   │   │   └── transformer/{model.py,decoder.py}
│   │   │   └── hybrid/                            [F]
│   │   ├── layers/
│   │   ├── heads/
│   │   ├── adapters/
│   │   └── noise_encoders/
│   │
│   ├── training/                                  [E/R]
│   │   ├── spec.py                                [N]
│   │   ├── execution.py                           [N]
│   │   ├── distributed/                           [N]
│   │   │   ├── ddp.py
│   │   │   ├── launcher.py
│   │   │   └── process_group.py
│   │   ├── trainers/
│   │   │   ├── base.py
│   │   │   ├── rbm.py
│   │   │   ├── supervised.py
│   │   │   └── multitask.py
│   │   ├── checkpoint/                            [N]
│   │   │   ├── model.py
│   │   │   ├── recovery.py
│   │   │   └── validation.py
│   │   ├── objectives/
│   │   ├── optimizers/
│   │   ├── schedulers/
│   │   ├── callbacks/
│   │   ├── curriculum/
│   │   ├── hard_mining/
│   │   └── regularization/
│   │
│   ├── evaluation/                                [N/R]
│   │   ├── scientific/
│   │   │   ├── evaluator.py
│   │   │   ├── protocol.py
│   │   │   ├── metrics.py
│   │   │   ├── logical_error_rate.py
│   │   │   ├── confidence.py
│   │   │   ├── calibration.py
│   │   │   ├── diagnostics.py
│   │   │   └── accuracy_gate.py
│   │   ├── baselines/
│   │   │   ├── registry.py
│   │   │   ├── mwpm.py
│   │   │   ├── exact_posterior.py
│   │   │   └── pymatching.py
│   │   └── regression/
│   │       ├── baseline.py
│   │       ├── historical.py
│   │       └── tolerance.py
│   │
│   ├── deployment/                                [E/R]
│   │   ├── runtime/
│   │   │   ├── protocol.py
│   │   │   ├── pytorch_cpu.py
│   │   │   ├── pytorch_cuda.py
│   │   │   ├── onnx.py                           [F]
│   │   │   └── tensorrt.py                       [F]
│   │   ├── profiling/
│   │   │   ├── latency.py
│   │   │   ├── throughput.py
│   │   │   ├── memory.py
│   │   │   └── telemetry.py                      [N]
│   │   ├── optimization/
│   │   │   ├── quantization.py
│   │   │   ├── pruning.py
│   │   │   ├── precision.py
│   │   │   └── export.py
│   │   ├── hardware/
│   │   │   ├── fpga/                             [F]
│   │   │   └── asic/                             [F]
│   │   ├── tradeoff/                              [F]
│   │   │   ├── candidate.py
│   │   │   ├── study.py
│   │   │   ├── pareto.py
│   │   │   └── constraints.py
│   │   ├── realtime/                              [F]
│   │   └── feedback/                              [F]
│   │
│   ├── reporting/                                 [N]
│   ├── visualization/                             [N]
│   ├── adaptation/                                [E/F]
│   ├── runtime/                                   [E/F]
│   ├── design/                                    [E/F]
│   └── utils/                                     [E/R]
│       ├── serialization.py
│       ├── hashing.py                             [N]
│       ├── paths.py                               [N]
│       ├── environment.py                         [N]
│       └── validation.py                          [N]
│
├── configs/                                       [E]
│   ├── experiment.smoke.yaml
│   ├── experiment.example.yaml
│   ├── qec/                                       [N]
│   ├── noise/                                     [N]
│   ├── models/                                    [N]
│   ├── training/                                  [N]
│   └── evaluation/                                [N]
│
├── scripts/                                       [E]
│   ├── run_experiment.py
│   ├── generate_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── export_model.py
│   ├── export_paper.py
│   ├── adapt.py
│   ├── check_openspec_governance.py
│   ├── recover_run.py                             [N]
│   ├── inspect_dataset.py                         [N]
│   ├── verify_run.py                              [N]
│   └── profile_decoder.py                         [N]
│
├── paper/                                         [E]
│   ├── docs/
│   ├── srcs/
│   │   ├── torlai_melko_2017.ipynb
│   │   └── <future-paper>.ipynb
│   ├── summary/
│   ├── templates/
│   └── releases/                                  [G]
│
├── tests/                                         [E/R]
│   ├── unit/
│   │   ├── experiment/
│   │   ├── data/
│   │   ├── qec/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── governance/
│   ├── integration/                               [N]
│   ├── smoke/
│   ├── acceptance/                                [N]
│   └── fixtures/
│
├── datasets/                                      [G]
│   ├── registry.json
│   └── <dataset-artifact-id>/
│       ├── manifest.json
│       ├── checksums.json
│       ├── generation.json
│       ├── train/
│       ├── validation/
│       └── test/
│
├── runs/                                          [G]
│   └── <experiment-id>/
│       ├── experiment.json
│       ├── resolved_plan.json
│       └── attempts/
│           ├── run-0001/
│           └── run-0002/
│
└── .github/                                       [optional]
    └── workflows/
        ├── tests.yml
        ├── openspec.yml
        └── notebook-smoke.yml
```

---

# 24. Runtime 文件布局

## DatasetArtifact

```text
datasets/
└── <dataset-artifact-id>/
    ├── manifest.json
    ├── checksums.json
    ├── generation.json
    ├── train/
    │   ├── shard-00000.*
    │   ├── shard-00001.*
    │   └── ...
    ├── validation/
    │   └── ...
    └── test/
        └── ...
```

## Experiment / Attempts

建议逻辑结构：

```text
runs/
└── <experiment-id>/
    ├── experiment.json
    ├── resolved_plan.json
    └── attempts/
        ├── run-0001/
        │   ├── run.json
        │   ├── config.yaml
        │   ├── environment.json
        │   ├── git.json
        │   ├── dataset_instance.json
        │   ├── stages/
        │   │   ├── dataset.json
        │   │   ├── training.json
        │   │   ├── scientific_evaluation.json
        │   │   ├── accuracy_gate.json
        │   │   ├── performance.json
        │   │   └── report.json
        │   ├── checkpoints/
        │   │   ├── model/
        │   │   │   ├── best.pt
        │   │   │   └── final.pt
        │   │   └── recovery/
        │   │       └── epoch-0037.pt
        │   ├── predictions/
        │   ├── metrics/
        │   ├── figures/
        │   ├── artifacts.json
        │   └── logs/run.log
        └── run-0002/
            ├── run.json
            ├── recovery_from.json
            └── ...
```

注意：实际是否迁移为 `<experiment-id>/attempts/` 物理路径，需要在 OpenSpec design 阶段考虑兼容现有 `runs/<run_id>/` 证据。可以先保留旧 layout，在 manifest 中增加 `experiment_id` / `attempt_id`。

---

# 25. `utils/` 收敛原则

当前部分 lifecycle 能力位于：

```text
utils/
├── experiment_setup.py
├── run_record.py
├── source_run.py
├── random_streams.py
├── artifact_integrity.py
├── config.py
├── reproducibility.py
└── serialization.py
```

长期领域归属应更接近：

```text
experiment/
├── experiment.py
├── run.py
├── recovery.py
├── random_streams.py
├── artifact.py
└── integrity.py
```

`utils/` 只保留真正通用的小工具。不要一次性机械搬文件；迁移必须保持 API compatibility 并有测试。

---

# 26. 推荐的第一轮 OpenSpec 子变更

不要一次做“重构整个项目”的超大 change。

## Change 1 — Enable cross-run DatasetArtifact reuse

目标：

```text
DatasetArtifact reusable
DatasetInstance run-bound
DatasetRegistry
DatasetKey
verified cache hit
cache miss generation
```

验收至少包括：

```text
same generation spec → same reusable artifact
second run does not regenerate
corrupt artifact → reject cache hit
different generation semantics → cache miss
model/training/execution change → does not invalidate dataset
```

## Change 2 — Add recoverable experiment attempts

目标：

```text
failed Attempt remains terminal
new Attempt recovery_from previous
safe stage recovery
training recovery checkpoint separated from model checkpoint
clean notebook Run All recovery
```

v0.1：epoch-boundary training recovery + stage-boundary experiment recovery。

## Change 3 — Add Scientific Accuracy Gate

目标：

```text
ScientificEvaluationSpec
AccuracyGateSpec
ScientificAcceptanceResult
same-dataset baseline comparison
statistical tolerance
```

Torlai–Melko notebook 是第一条真实 acceptance path。

## Change 4 — Separate TrainingSpec and ExecutionSpec

目标：

```text
TrainingSpec
ExecutionSpec
single GPU
multi-GPU DDP
AMP
workers
compile
```

验收：

```text
ExecutionSpec change does not change DatasetKey
DDP config appears in ResolvedPlan
unsupported execution options fail before run/data creation
```

---

# 27. 实施优先级

```text
P0
├── Dataset reuse
├── Experiment / Notebook recovery
└── Scientific Accuracy Gate

P1
├── TrainingSpec / ExecutionSpec split
├── DDP
├── generic neural training path
└── stronger integration tests

P2
├── GPU inference benchmark
├── TensorRT / ONNX adapters
└── standardized performance reports

P3
├── FPGA simulation / synthesis
├── ASIC simulation / synthesis
└── Accuracy–Performance Pareto study
```

---

# 28. v0.1 明确不做

除非用户明确改变范围，不要在近期核心变更中主动引入：

```text
Kubernetes
Ray cluster orchestration
Ray Data
DeepSpeed
Tensor Parallel
Pipeline Parallel
FSDP2
Triton Server
full realtime QPU feedback
FPGA board implementation
ASIC physical design flow
huge visualization framework
database-heavy MLOps platform
```

---

# 29. 第一条平台 Acceptance Path

使用：

```text
paper/srcs/torlai_melko_2017.ipynb
```

最终至少覆盖：

```text
1. clean kernel Run All succeeds
2. Dataset cache miss generates valid artifact
3. Dataset cache hit reuses same verified artifact
4. repeated training uses same DatasetArtifact
5. training crash leaves old Attempt terminal
6. recovery creates new Attempt
7. recovery restores from safe checkpoint
8. scientific evaluation runs AI + MWPM on same test data
9. LER / CI are produced
10. Accuracy Gate is persisted
11. artifacts remain immutable
12. run finalization verifies hashes
```

Notebook smoke 仍不能被解读为论文规模科学复现结果。

---

# 30. 测试架构

```text
tests/
├── unit/
│   ├── experiment/
│   ├── data/
│   ├── qec/
│   ├── training/
│   ├── evaluation/
│   └── governance/
├── integration/
│   ├── test_dataset_train_evaluate.py
│   ├── test_dataset_reuse_across_runs.py
│   ├── test_checkpoint_model_reuse.py
│   ├── test_training_recovery.py
│   └── test_accuracy_gate_pipeline.py
├── smoke/
│   ├── test_pipeline.py
│   └── test_notebook_smoke.py
├── acceptance/
│   ├── test_torlai_clean_run.py
│   ├── test_torlai_dataset_cache_hit.py
│   ├── test_torlai_training_crash_recovery.py
│   ├── test_torlai_evaluation_recovery.py
│   └── test_torlai_baseline_regression.py
└── fixtures/
```

测试语义：

```text
unit        → contract / pure behavior
integration → multi-component semantics
smoke       → executable path
acceptance  → OpenSpec scenario / real paper workflow
```

---

# 31. Codex 实施纪律

Codex 收到“设计”请求时：

```text
1. inspect current OpenSpec
2. create/update bounded change
3. write proposal/spec/design/tasks
4. strict validate
5. stop before code unless implementation was explicitly authorized
```

收到“实现/开发/修复/重构”请求时：

```text
1. inspect current specs/change
2. create/update bounded change if necessary
3. complete proposal/spec/design/tasks
4. strict validate
5. implement
6. run tests
7. run OpenSpec verification
8. write verification.md
9. update capability evidence only when justified
10. archive only when gates pass
```

不要：

```text
- 未改 requirement 就改变 validated semantics
- 为了目录整洁做无意义大搬家
- 把 scaffold 当成 capability 已实现
- 静默 fallback backend
- 把 performance estimate 说成 measurement
- 修改已登记 immutable artifact
- 把 failed Run 原地改成 successful resumed Run
```

---

# 32. 目标目录的解释规则

本文件中的目标目录不是要求：

```text
mkdir 所有 Future 文件
```

而是：

> 当对应 capability 真正进入实现时，将代码放入明确的 domain boundary。

例如 FPGA 尚未进入正式实现，就不要生成大量空 `fpga/*.py` scaffold。CNN / GNN 尚未使用，也不要为了满足目录树创建空文件。

---

# 33. 最近期的最小目标代码边界

如果当前只实现近期需求，优先收缩为：

```text
ai_qec/
├── experiment/
│   ├── run.py
│   ├── stage.py
│   ├── recovery.py
│   ├── artifact.py
│   └── random_streams.py
├── data/datasets/
│   ├── spec.py
│   ├── identity.py
│   ├── artifact.py
│   ├── instance.py
│   ├── registry.py
│   └── resolution.py
├── training/
│   ├── spec.py
│   ├── execution.py
│   └── checkpoint/
│       ├── model.py
│       └── recovery.py
└── evaluation/
    └── scientific/
        ├── evaluator.py
        ├── metrics.py
        └── accuracy_gate.py
```

这四块直接服务：

```text
Dataset reuse
+
Notebook crash recovery
+
AI vs traditional scientific baseline
+
Multi-GPU execution
```

其他未来能力不要抢先实现。

---

# 34. 最终架构判断标准

任何新组件、依赖或目录，都先回答：

> **它插在 Golden Path 的哪个位置？**

```text
Configure
   ↓
Dataset Resolve / Reuse
   ↓
AI Training
   ↓
Scientific Evaluation
   ↓
Accuracy Gate
   ↓
Performance Research
   ↓
Accuracy Revalidation
   ↓
Trade-off
   ↓
Report
```

如果无法回答，就暂时不要加入核心架构。

---

# 35. 一句话总结

> **AI-for-QEC 应从现有“可追溯论文复现实验框架”，演进为一个 Accuracy-first、Performance-second 的 QEC AI 研究平台：可复用不可变数据、可恢复实验执行、同协议经典基线比较、显式 Accuracy Gate，以及通过 Gate 后的 GPU / FPGA / ASIC 性能与 Accuracy–Performance Trade-off 研究。**
