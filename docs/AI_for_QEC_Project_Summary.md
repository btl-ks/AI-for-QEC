# AI for QEC 研究平台总结

## 1. 项目定位

目标是构建一套面向 **AI for Quantum Error Correction（AI for QEC）** 的长期研究基础设施，支持从论文复现、QEC 数据生成、AI 解码器训练与科学评估，到后续 GPU / FPGA / ASIC 性能优化。

总体原则：

- **Accuracy first, performance second**
- **Notebook 负责编排，公共 Python 包负责实现**
- **NVIDIA-first，但核心实验语义保持 vendor-neutral**
- **能复用成熟基础设施就不重复造轮子**
- **OpenSpec 管正式需求，实验平台负责实现能力**
- **失败 Run 不原地复活；Experiment 可创建新的 Attempt 继续恢复**

---

# 2. 当前黄金主路径（Golden Path）

```text
1. Configure
      ↓
2. Dataset Resolve / Reuse
      ↓
3. Train AI Decoder
      ↓
4. Scientific Accuracy Evaluation
      ↓
   Traditional Baseline
      ↓
   Accuracy Gate
      ├── FAIL → 返回模型 / 数据 / 训练研究
      └── PASS → Freeze Accuracy Baseline Model
                         ↓
5. Performance Optimization
      ├── GPU 软件优化
      ├── FPGA / ASIC 仿真与加速
      └── Accuracy / Performance Trade-off
                         ↓
6. Visualization / Research Report
```

这条路径是当前项目最高层的实验主干。

---

# 3. Phase I：准确率 / 科学正确性优先

## 3.1 实验参数定义

一次实验由以下规格组成：

```text
ExperimentSpec
|
+-- QECSpec
|   +-- code family
|   +-- distance
|   +-- rounds
|   +-- logical basis
|   +-- circuit / syndrome extraction
|
+-- NoiseSpec
|   +-- noise model
|   +-- physical error rate
|   +-- gate / measurement / idle error
|   +-- correlated noise
|
+-- DatasetSpec
|   +-- generator
|   +-- backend
|   +-- train / validation / test samples
|   +-- seed / random stream
|   +-- data format
|   +-- preprocessing
|
+-- ModelSpec
|   +-- CNN / GNN / Transformer / RBM / ...
|   +-- architecture parameters
|
+-- TrainingSpec
|   +-- optimizer
|   +-- learning rate
|   +-- epochs
|   +-- batch size
|   +-- scheduler
|   +-- checkpoint policy
|
+-- ExecutionSpec
|   +-- CPU / GPU
|   +-- GPU count
|   +-- DDP
|   +-- AMP
|   +-- num_workers
|   +-- torch.compile
|
+-- ScientificEvaluationSpec
    +-- baseline decoder
    +-- metrics
    +-- noise / distance sweep
    +-- regression criteria
```

---

# 4. Dataset 生命周期与复用

## 4.1 三层对象

不要把“数据定义”“实际数据”“某次 Run 使用的数据”混成一个对象。

```text
DatasetSpec
    ↓
DatasetArtifact
    ↓
DatasetInstance
    ↓
Run
```

### DatasetSpec

描述：

- QEC / noise 条件
- generator 与版本
- backend
- shots
- seeds / random streams
- split policy
- preprocessing
- schema / format

### DatasetArtifact

表示真正生成并可复用的不可变数据：

- immutable
- content hash
- manifest
- physical path
- integrity information
- train / validation / test

### DatasetInstance

表示“某次 Run 如何使用某个 DatasetArtifact”：

- run_id
- dataset_artifact_id
- role
- usage metadata

这样可以同时满足：

```text
DatasetArtifact 可跨多个训练 Run 复用
DatasetInstance 与 Run 保持明确绑定
```

## 4.2 Resolve / Reuse

```text
DatasetSpec
    ↓
Dataset Identity
    ↓
DatasetRegistry
    |
    +-- cache hit
    |      ↓
    |   verify artifact
    |      ↓
    |   reuse
    |
    +-- cache miss
           ↓
        generate
           ↓
        validate
           ↓
        commit immutable DatasetArtifact
           ↓
        register
```

### Dataset identity 应包含

- QECSpec identity
- NoiseSpec identity
- generator + version
- shots
- seed/random-stream definition
- split policy
- preprocessing
- schema/data format

### 不应参与 Dataset identity

- ModelSpec
- TrainingSpec
- GPU 数量
- DDP
- AMP
- torch.compile
- visualization settings

因此不同模型和不同训练超参数可以共享同一份 syndrome 数据。

---

# 5. QEC 数据生成

支持多个生成后端：

```text
QECSpec + NoiseSpec
        ↓
   Generator Adapter
        |
        +-- CPU
        |   +-- Stim
        |   +-- Sinter
        |
        +-- GPU
        |   +-- CUDA-Q QEC
        |   +-- cuStabilizer
        |
        +-- QPU / Hardware
            +-- Qiskit
            +-- CUDA-Q backend
```

统一输出：

```text
QECBatch
|
+-- syndrome
+-- target / logical observable
+-- physical error / measurement history
+-- detector coordinates
+-- round mask
+-- sample identity
+-- dataset identity
+-- provenance
```

模型不应依赖数据来自 Stim、CUDA-Q 还是 QPU。

---

# 6. 数据到 AI 的转换

默认优先使用成熟基础设施，不自行实现基础 queue runtime。

## v0.1

```text
Stim CPU workers
      ↓
PyTorch IterableDataset
      ↓
DataLoader
      ↓
prefetch
      ↓
pinned memory
      ↓
non-blocking H2D
      ↓
GPU training
```

主要参数：

- `num_workers`
- `prefetch_factor`
- `persistent_workers`
- `pin_memory`
- `non_blocking=True`

## 后续性能瓶颈出现时

可替换为：

- NVIDIA DALI
- GPU-native CUDA-Q / cuStabilizer path
- Ray Data（单机当前不是优先项）

---

# 7. AI Training

核心框架：

```text
PyTorch
```

模型保持原生：

```python
class Decoder(torch.nn.Module):
    ...
```

研究模型包括：

- CNN
- 3D CNN
- GNN
- Transformer
- Recurrent Transformer
- RBM
- Hybrid neural / classical decoder

训练能力：

```text
single GPU
    ↓
torchrun + DDP
    ↓
AMP
    ↓
torch.compile [optional]
    ↓
checkpoint
```

当模型单卡放不下时再考虑：

- FSDP2
- TP / PP
- DeepSpeed

当前不是优先项。

---

# 8. Scientific Accuracy Evaluation

训练后首先回答：

> AI decoder 的科学正确性是否达到传统算法基线水平？

主要比较：

```text
same test dataset
|
+-- AI Decoder
|
+-- Traditional Baseline
    +-- MWPM
    +-- PyMatching
    +-- Tesseract
    +-- CUDA-Q QEC decoder
    +-- paper-specific baseline
```

## 主指标

QEC 场景中优先使用：

- **Logical Error Rate (LER)** — primary
- decoder accuracy — secondary
- confidence interval
- failure distribution
- code-distance / noise generalization

不要只用普通 classification accuracy 作为最终科学指标。

---

# 9. Accuracy Gate

性能优化前设置明确门槛：

```text
AI Scientific Evaluation
        ↓
Accuracy Gate
    /          \
 FAIL          PASS
  ↓             ↓
继续模型研究     Freeze Golden Model
                ↓
         Performance Research
```

Gate 不建议简单使用：

```text
AI accuracy >= baseline accuracy
```

而应使用：

- 同一 test dataset
- 同一 QEC / noise protocol
- LER
- confidence interval
- 允许的 tolerance
- 预注册的判定规则

通过后产出：

```text
AccuracyBaselineModel / Golden Model
```

后续所有性能优化版本与该模型比较。

---

# 10. Phase II：Performance Optimization

只有 Accuracy Gate 通过后，才正式进入性能优化。

分成两条主线。

## 10.1 FPGA / ASIC 仿真与加速

```text
Golden PyTorch Model
        ↓
Quantized Representation
        ↓
FPGA / ASIC Mapping
        ↓
Simulation / Synthesis
        ↓
Golden Verification
        ↓
Performance Report
```

### FPGA 可能工具

- Brevitas
- FINN
- hls4ml
- AMD Vitis HLS
- Verilator
- cocotb

指标：

- latency
- throughput
- LUT
- FF
- DSP
- BRAM
- power

### ASIC / IC

路线：

```text
PyTorch golden model
        ↓
quantized / fixed-point specification
        ↓
HLS / RTL
        ↓
RTL simulation
        ↓
synthesis
        ↓
timing / area / power
```

后续根据实验室工具链选择：

- Synopsys
- Cadence
- Siemens
- OpenROAD
- Verilator

---

# 11. Accuracy / Performance Trade-off

性能阶段不只追求最快，也不只追求最低 LER。

需要比较：

```text
Candidate A
LER 很低
latency 高

Candidate B
LER 稍高
latency 中

Candidate C
LER 接近 baseline
latency 极低
```

主要优化手段：

- quantization
- pruning
- smaller model
- reduced precision
- fewer layers
- fewer decoder iterations
- early exit
- hybrid AI + classical decoding

最终目标：

```text
Pareto Frontier
```

即寻找在 Accuracy / Latency / Power / Hardware Cost 等指标上不可被同时支配的方案。

---

# 12. Scientific Evaluation 与 Performance Evaluation 分离

## ScientificEvaluationSpec

负责：

- LER
- baseline
- physical error sweep
- code distance
- confidence interval
- generalization
- Accuracy Gate

## PerformanceEvaluationSpec

负责：

- p50 latency
- p95 latency
- p99 latency
- throughput
- memory
- power
- GPU utilization
- FPGA resources
- ASIC area / timing / power
- accuracy degradation
- trade-off analysis

这样可以避免“模型没学好”和“实现不够快”混为一个问题。

---

# 13. Notebook 论文复现模式

当前论文复现 Notebook 的定位：

```text
Notebook
|
+-- paper metadata
+-- configuration
+-- stage ordering
+-- paper-specific protocol
+-- visualization
+-- interpretation
|
| calls
v
Shared ai_qec package
|
+-- experiment lifecycle
+-- dataset
+-- QEC
+-- noise
+-- training
+-- decoder
+-- baseline
+-- metrics
+-- artifacts
+-- recovery
```

Notebook 只做 orchestration，不承担核心实现。

公共入口建议保持：

```python
import ai_qec.notebook_api as qec
```

避免历史论文 notebook 直接依赖大量内部模块。

---

# 14. 标准论文实验阶段

```text
Paper Notebook
|
+-- 1. Configure
|
+-- 2. Resolve / Start Run
|
+-- 3. Resolve Dataset
|   +-- reuse?
|   +-- generate?
|
+-- 4. Train AI
|
+-- 5. Scientific Evaluate
|
+-- 6. Traditional Baseline
|
+-- 7. Accuracy Gate
|
+-- 8. Performance Evaluation [PASS 后]
|
+-- 9. Visualization / Report
|
+-- 10. Finish
```

示意 API：

```python
CONFIG = {
  "qec": {
        "code_family": "surface",
        "distance": 3,
        "rounds": 3,
        "logical_basis": "Z"
      },
    'generator' = 'stim',
    'trainer' = 'cuda',
    ...
}

experiment = qec.create_experiment(CONFIG)

run = experiment.start_or_recover()

dataset = run.resolve_dataset()

model = run.train(dataset)

scientific_result = run.evaluate_accuracy(
    model=model,
    dataset=dataset,
    baselines=["mwpm"],
)

if scientific_result.accuracy_gate_passed:
    performance_result = run.evaluate_performance(
        model=model,
        dataset=dataset,
    )

run.visualize()
run.finish()
```

---

# 15. Notebook 异常恢复

恢复单位应是 **Experiment Stage**，不是 Notebook Cell。

```text
Experiment E001
|
+-- Attempt R001
|   +-- configure        COMPLETE
|   +-- dataset          COMPLETE
|   +-- training         FAILED
|   +-- evaluation       NOT_STARTED
|
+-- Attempt R002
    +-- recovery_from = R001
    +-- configuration    verify
    +-- dataset          reuse
    +-- training         checkpoint/recover
    +-- evaluation
    +-- finish
```

核心规则：

- Failed Run 保持 FAILED
- 不原地修改失败历史
- 恢复创建新的 Attempt
- 新 Attempt 显式引用恢复来源
- model checkpoint 与 execution recovery state 分离

---

# 16. Random Streams

不只保存一个 `seed=42`。

至少区分：

```text
RandomStreams
|
+-- qec_sampling
+-- dataset_split
+-- model_init
+-- training_shuffle
+-- adaptive_sampling
+-- decoder_sampling
```

要求：

- named
- independent
- deterministic reconstruction
- recovery-compatible

---

# 17. Artifact & Evidence

所有关键输出作为 Artifact：

```text
Artifact
|
+-- artifact_id
+-- type
+-- producer_run
+-- producer_stage
+-- path
+-- checksum
+-- metadata
```

典型 artifacts：

- DatasetArtifact
- model checkpoint
- best model
- predictions
- metrics
- benchmark reports
- figures
- ONNX
- TensorRT engine
- FPGA/ASIC simulation outputs

正式结果必须可独立复核。

---

# 18. Testing

建议分层：

```text
Unit
    ↓
Smoke
    ↓
Integration
    ↓
Scientific Regression
    ↓
Performance Regression
    ↓
Hardware Verification
```

当前优先测试：

1. `QECSpec / NoiseSpec / DatasetSpec`
2. Dataset identity
3. Dataset reuse
4. Random stream reproducibility
5. Artifact checksum
6. Notebook clean `Run All`
7. Dataset cache miss path
8. Dataset cache hit path
9. training crash → recovery
10. evaluation crash → recovery
11. AI vs traditional baseline
12. Accuracy Gate

---

# 19. OpenSpec 能力架构

保留原有正式 Capability 结构：

```text
AI for QEC Requirements
|
+-- 1. Governance
|   +-- governance/spec-management
|
+-- 2. Research
|   +-- research/experiment-lifecycle
|   +-- research/paper-reproduction
|   +-- research/data-efficient-training
|   +-- research/automated-validation
|
+-- 3. QEC Core
|   +-- qec/dataset-pipeline
|   +-- qec/decoder-workbench
|
+-- 4. Deployment
    +-- deployment/performance-runtime
```

未来可能新增：

```text
qec/ftqc-runtime
qec/design-automation
deployment/realtime-control
```

但当前不需要提前加入正式 scope。

---

# 20. 当前最重要的需求调整

原有：

```text
中断执行不得断点续跑
Checkpoint 不携带执行恢复状态
```

调整为：

```text
失败 Run SHALL 进入明确终态。

Experiment MAY 创建新的 Attempt，
并从经过验证的安全恢复点继续执行。

新的 Attempt SHALL 显式引用 recovery source。

Model checkpoint 与 execution recovery state SHALL 分离。
```

---

# 21. 当前真正需要稳定的核心对象

近期核心对象：

```text
ExperimentSpec
Experiment

Run / Attempt
Stage
RecoveryState

RandomStreams

QECSpec
NoiseSpec

DatasetSpec
DatasetArtifact
DatasetInstance
DatasetRegistry
QECBatch

ModelArtifact

DecodeRequest
DecodeResult

ScientificEvaluationSpec
PerformanceEvaluationSpec
AccuracyGate

Artifact
ArtifactManifest

PerformanceReport
```

第三方工具均作为这些对象下面的实现。

---

# 22. 推荐技术栈

## QEC / Data

```text
Stim / Sinter
CUDA-Q QEC
cuStabilizer
Qiskit [QPU]
PyMatching
Tesseract
```

## AI

```text
PyTorch
PyTorch Geometric [GNN]
torchrun
DDP
AMP
torch.compile
```

## Data Pipeline

```text
PyTorch DataLoader
NVIDIA DALI [performance need]
```

## Storage

```text
Zarr / HDF5
Parquet / JSON manifest
SQLite / DuckDB registry
```

## Deployment

```text
ONNX
ONNX Runtime
TensorRT
```

## Performance

```text
torch.profiler
Nsight Systems
Nsight Compute
DCGM / NVML
```

## FPGA / ASIC

```text
Brevitas
FINN
hls4ml
Vitis HLS
Verilator
cocotb
```

## Experiment / Quality

```text
OpenSpec
pytest
Hypothesis
pytest-benchmark
MLflow
Git
Docker
```

---

# 23. 最终研究主线

```text
QEC / Noise / Model Configuration
                ↓
        Dataset Resolve / Reuse
                ↓
             AI Train
                ↓
     Scientific Accuracy Evaluation
                ↓
        Traditional Baseline
                ↓
          ACCURACY GATE
          /           \
       FAIL            PASS
        ↓               ↓
Model Research      Freeze Model
                        ↓
                Performance Research
                    /          \
          FPGA / ASIC         Accuracy /
          Acceleration        Performance
                              Trade-off
                    \          /
                     ↓        ↓
                   Pareto Analysis
                         ↓
                  Visualization
                         ↓
                  Research Result
```

---

# 24. 当前开发优先级

## P0 — 近期必须完成

- Experiment / Run / Stage lifecycle
- `.ipynb` clean Run All
- crash → new Attempt recovery
- RandomStreams
- DatasetSpec
- DatasetArtifact
- DatasetInstance
- DatasetRegistry
- dataset reuse
- QECBatch
- PyTorch training
- traditional baseline
- ScientificEvaluation
- AccuracyGate
- Artifact integrity
- unit / smoke / recovery tests

## P1 — Accuracy Research

- CNN / GNN / Transformer
- QEC/noise sweep
- baseline regression
- generalization
- data-efficient training
- automated validation

## P2 — Performance Research

只有 Accuracy Gate 通过后正式开展：

- GPU profiling
- TensorRT
- quantization
- FPGA simulation
- ASIC simulation
- accuracy / performance trade-off
- Pareto analysis

## P3 — Future

- realtime QEC
- NVQLink
- QPU feedback
- FTQC runtime
- QEC code / architecture search

---

## 核心原则

> **先证明 AI decoder 在科学指标上达到传统算法基线，再研究低时延和硬件加速。**

> **自己维护 AI-for-QEC 的实验语义、数据语义、科学验证和恢复契约；底层计算、GPU、分布式、profiling 与硬件工具链尽量复用成熟组件。**
