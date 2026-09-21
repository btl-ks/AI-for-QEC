# AI for QEC Research Platform — System Architecture

## 1. 系统目标

该平台面向长期 AI for Quantum Error Correction 研究，目标是建立一套可复现、可扩展、可验证的实验基础设施，覆盖：

```text
QEC / Noise 定义
        ↓
量子电路与 Syndrome 数据生成
        ↓
数据标准化、缓存与复用
        ↓
CPU / GPU 数据流水线
        ↓
AI Decoder 训练
        ↓
Classical / AI / Hybrid Decoder
        ↓
CPU / GPU / TensorRT / FPGA / ASIC 部署
        ↓
性能、科学正确性与硬件一致性验证
```

平台采用：

```text
NVIDIA-first
Vendor-neutral core
```

即优先利用 NVIDIA CUDA-Q、cuStabilizer、TensorRT、Nsight 等成熟基础设施，但核心研究接口不绑定任何单一厂商。

---

# 2. Module 1 — Specs & Management

## 职责

定义：

```text
研究什么
生成什么数据
使用什么 noise
训练什么模型
如何评价实验
```

## 组件

### OpenSpec

负责：

```text
requirements
change proposal
design
acceptance criteria
implementation tasks
change history
```

选型：

```text
Default:
OpenSpec

Alternative:
GitHub Issues
Jira
Linear
```

### QECSpec

负责描述 QEC 本身：

```text
code family
distance
rounds
logical basis
logical operations
syndrome extraction circuit
measurement semantics
```

例如：

```text
Surface Code
distance = 7
rounds = 20
logical_basis = Z
```

推荐：

```text
Pydantic model
```

此接口由项目自己维护。

### NoiseSpec

负责统一描述 noise：

```text
noise family
error probability
gate noise
measurement noise
reset noise
idle noise
correlation
time dependence
```

例如：

```text
NoiseSpec

1Q depolarizing = 1e-4
2Q depolarizing = 1e-3
measurement = 2e-3
```

再通过：

```text
NoiseCompiler
```

转为：

```text
Stim noise
CUDA-Q NoiseModel
Qiskit noise
```

禁止 backend 做未记录的隐式 approximation。

### DatasetSpec

定义：

```text
number of shots
train / validation / test split
seed range
storage representation
generator
generator version
```

### ExperimentSpec

组合一次完整实验：

```text
ExperimentSpec
├── QECSpec
├── NoiseSpec
├── DatasetSpec
├── ModelSpec
├── TrainingSpec
├── RuntimeSpec
└── VerificationSpec
```

推荐：

```text
Pydantic
YAML
Hydra（后期做大量 sweep）
```

### Version Control

负责：

```text
source code
config
OpenSpec
schema
experiment implementation
```

推荐：

```text
Git
GitHub / GitLab
```

---

# 3. Module 2 — Circuit & Noise Adapter Layer

## 职责

把项目内部的：

```text
QECSpec
NoiseSpec
```

转换为具体平台可以执行的形式。

架构：

```text
QECSpec + NoiseSpec
        ↓
     Adapter
   ┌────┼─────┐
   ↓    ↓     ↓
 Stim CUDA-Q Qiskit
```

## 组件

### StimAdapter

负责：

```text
QECSpec
NoiseSpec
    ↓
Stim Circuit
Detector Error Model
Stim sampler
```

用途：

```text
CPU reference
高速 syndrome generation
scientific baseline
```

### CUDA-Q Adapter

负责：

```text
QECSpec
NoiseSpec
    ↓
CUDA-Q
CUDA-Q QEC
cuStabilizer
```

用途：

```text
GPU accelerated simulation
GPU syndrome generation
QEC decoders
realtime integration
```

### QiskitAdapter

负责：

```text
circuit construction
circuit visualization
IBM backend
real QPU data
```

定位：

```text
Hardware adapter
```

而不是平台主要 simulator。

### NoiseCompiler

负责检查：

```text
NoiseSpec
       ↓
backend capability
       ↓
exact?
 /   \
yes   no
      ↓
 explicit approximation
 or reject
```

例如：

```text
Amplitude damping
        ↓
Stim
        ↓
unsupported exactly
```

平台应该报错或者要求显式指定：

```text
Pauli twirling approximation
```

不能偷偷转换。

---

# 4. Module 3 — Syndrome Data Generation

## 职责

产生 AI decoder 所需训练/验证/测试数据。

平台支持两个主要 Data Plane。

## CPU Data Plane

```text
QECSpec
NoiseSpec
   ↓
Stim / Sinter
   ↓
many-core CPU
   ↓
syndrome
logical observable
error information
```

推荐：

```text
Stim
Sinter
```

适合：

```text
CPU utilization
large many-core server
reference datasets
large offline generation
```

## GPU-native Data Plane

```text
QECSpec
NoiseSpec
   ↓
CUDA-Q QEC
cuStabilizer
   ↓
NVIDIA GPU
   ↓
CUDA syndrome tensor
   ↓
PyTorch
```

推荐：

```text
CUDA-Q QEC dem_sampling
cuStabilizer
```

优势：

```text
GPU generation
GPU resident data
avoid unnecessary host-device copy
high-throughput sampling
```

## Hardware Data Plane

未来：

```text
IBM / other QPU
       ↓
Qiskit / CUDA-Q backend
       ↓
measurement data
       ↓
QECBatch
```

## Seed / Sharding Manager

负责：

```text
seed allocation
worker shard
sample identity
duplicate prevention
resume position
```

例如：

```text
worker 0:
seed 0 - 9999

worker 1:
seed 10000 - 19999
```

确保：

```text
no duplicate
no missing shard
reproducible generation
```

---

# 5. Module 4 — Data Pipeline & Dataset Management

这是系统非常核心的一层。

## QECBatch

所有 backend 最终必须转换为统一数据结构：

```text
QECBatch

syndrome
target / logical observable
measurements
errors
detector coordinates
round mask

sample_id
dataset_id

QEC metadata
Noise metadata
generator metadata
```

架构：

```text
Stim
       \
CUDA-Q ───→ QECBatch → AI model
       /
QPU
```

模型不应该知道数据来自哪个 backend。

## Tensorizer

负责：

```text
persistent representation
        ↓
model representation
```

例如：

```text
bit-packed uint8
        ↓
bool tensor
        ↓
float16 / float32
```

可选技术：

```text
NumPy
PyTorch
CuPy
DLPack
```

默认：

```text
NumPy + PyTorch
```

---

# 6. Dataset Storage

建议将：

```text
数据本体
```

和：

```text
metadata
```

分开。

数据本体：

```text
Zarr
HDF5
memory map
packed binary
```

Metadata：

```text
JSON
YAML
Parquet
```

推荐组合：

```text
Zarr
+
Parquet / JSON manifest
```

不要长期把 syndrome 保存为：

```text
float32 tensor
```

优先：

```text
bit-packed
uint8
```

---

# 7. DatasetRegistry

负责：

```text
这份数据以前生成过吗？
```

以及：

```text
在哪里？
哪个 noise？
哪个 code？
多少 shots？
谁生成的？
什么版本？
```

身份体系：

```text
QECID
    = hash(QECSpec)

NoiseID
    = hash(NoiseSpec)

GeneratorID
    = hash(generator + version)

DatasetID
    = hash(
        QECID
        + NoiseID
        + GeneratorID
        + shots
        + split
        + seed
        + schema version
      )
```

如果：

```text
DatasetID exists
```

则：

```text
reuse
```

否则：

```text
generate
→ store
→ register
```

实现可选：

```text
v0.1:
filesystem + manifest

v0.2:
SQLite / DuckDB

larger collaboration:
DVC / lakeFS
```

---

# 8. CPU → GPU Streaming Pipeline

v0.1 不自己造 queue runtime。

默认：

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

PyTorch 使用：

```text
num_workers
prefetch_factor
persistent_workers
pin_memory
non_blocking=True
```

如果 profiling 表明 pipeline 成为瓶颈：

```text
PyTorch DataLoader
        ↓
NVIDIA DALI
```

DALI 可以提供：

```text
CPU pipeline
GPU pipeline
async execution
prefetch
queue
CPU/GPU overlap
```

可选：

```text
Ray Data
```

但在你的单服务器环境里不是第一选择。

---

# 9. Module 5 — AI Training

核心框架：

```text
PyTorch
```

模型保持：

```python
class Decoder(torch.nn.Module):
    ...
```

不要再定义一套自有 Model Framework。

## Model Families

研究模型可以包括：

```text
MLP

CNN
3D CNN

GNN

Transformer

Recurrent Transformer

Hybrid neural/classical model
```

GNN 可选：

```text
PyTorch Geometric
DGL
```

推荐：

```text
PyTorch + PyG
```

---

# 10. Training Runtime

单 GPU：

```text
PyTorch
```

多 GPU：

```text
torchrun
+
DistributedDataParallel
```

如果未来模型一张 GPU 放不下：

```text
FSDP2
```

暂时没有必要优先考虑：

```text
DeepSpeed
Tensor Parallel
Pipeline Parallel
```

## Mixed Precision

推荐：

```text
torch.autocast
AMP
```

性能进一步优化：

```text
torch.compile
```

---

# 11. Checkpoint & Fault Recovery

Checkpoint 至少保存：

```text
model
optimizer
scheduler
AMP scaler
global step
RNG
dataset cursor
dataset_id
experiment_id
```

单 GPU：

```text
torch.save
```

多 GPU：

```text
Distributed Checkpoint
```

进程恢复：

```text
torchrun restart
```

---

# 12. Classical Decoder Baseline

必须长期保留 classical baseline。

推荐：

```text
PyMatching
Tesseract
CUDA-Q QEC decoders
```

CUDA-Q QEC 还可以接：

```text
Fusion decoder
QLDPC decoder
Tensor Network decoder
Chromobius
LUT
Sliding-window decoder
```

---

# 13. Hybrid Decoder

支持：

```text
syndrome
   ↓
AI predecoder GPU
   ↓
resolved?
 /      \
yes      no
 ↓        ↓
result   CPU classical decoder
           ↓
         result
```

例如：

```text
Transformer/CNN
      +
PyMatching
```

然后比较：

```text
AI only
classical only
hybrid
```

指标：

```text
LER
latency
p99
throughput
CPU utilization
GPU utilization
```

---

# 14. Module 6 — Decoder Runtime

训练模型与部署 runtime 分开。

统一接口：

```text
DecoderRuntime
```

概念接口：

```python
decode(QECBatch) -> DecodeResult
```

实现：

```text
TorchCPUDecoder

TorchCUDADecoder

ONNXDecoder

TensorRTDecoder

FPGADecoder

ASICSimulatorDecoder

ClassicalDecoder
```

研究代码只调用：

```text
decoder.decode()
```

不直接依赖具体硬件。

---

# 15. ONNX

用途：

```text
model interchange
deployment boundary
cross-backend comparison
```

路线：

```text
PyTorch
   ↓
ONNX
   ├─ ONNX Runtime
   ├─ TensorRT
   ├─ FPGA tool
   └─ ASIC flow
```

---

# 16. TensorRT

主要 NVIDIA GPU deployment runtime。

用途：

```text
FP32
FP16
BF16
INT8
```

路线：

```text
PyTorch
   ↓
ONNX
   ↓
TensorRT
   ↓
GPU decoder
```

可选：

```text
Torch-TensorRT
```

---

# 17. FPGA

后期支持。

可能工具：

```text
Brevitas
FINN
hls4ml
AMD Vitis HLS
Intel HLS
```

适合研究：

```text
quantization
fixed point
low latency
resource utilization
```

输出指标：

```text
latency
throughput
LUT
FF
DSP
BRAM
power
```

---

# 18. ASIC / IC

后期。

路线：

```text
PyTorch golden model
        ↓
quantized representation
        ↓
HLS / RTL
        ↓
RTL simulation
        ↓
synthesis
        ↓
timing / power / area
```

工具可能：

```text
Synopsys
Cadence
Siemens
OpenROAD
Verilator
```

---

# 19. Golden Verification

必须有统一：

```text
GoldenVectorSet
```

同一批 syndrome：

```text
PyTorch CPU
PyTorch CUDA
ONNX
TensorRT
FPGA simulation
ASIC simulation
```

比较：

```text
raw output
decoder decision
logical error rate
numerical error
```

这样硬件迁移不会出现：

```text
能跑
但 scientific result 已经变了
```

却没人发现。

---

# 20. Module 7 — Evaluation

科学指标：

```text
Logical Error Rate
decoder accuracy
threshold
generalization
```

系统指标：

```text
latency

p50
p95
p99

throughput

memory usage

CPU utilization

GPU utilization

power
```

硬件指标：

```text
FPGA:
LUT / FF / BRAM / DSP

ASIC:
area / timing / power
```

---

# 21. Testing Framework

统一使用：

```text
pytest
```

再配：

```text
Hypothesis
pytest-benchmark
```

测试分层：

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

## Unit Test

测试：

```text
QECSpec
NoiseSpec
QECBatch
DatasetID
serialization
schema validation
```

## Smoke Test

几十秒跑完整链：

```text
d=3
small noise
128 shots
   ↓
Stim
   ↓
QECBatch
   ↓
PyTorch
   ↓
train 2 steps
   ↓
checkpoint
   ↓
reload
   ↓
decode
   ↓
PASS
```

## Scientific Regression

测试：

```text
LER
noise sweep behavior
reference baseline
distance scaling
statistical properties
```

## Performance Regression

防止：

```text
新 commit
导致 throughput ↓30%
```

或者：

```text
H2D latency ↑2×
```

---

# 22. Profiling

分四层。

Framework：

```text
torch.profiler
```

系统 timeline：

```text
Nsight Systems
```

CUDA kernel：

```text
Nsight Compute
```

服务器 GPU telemetry：

```text
DCGM
NVML
nvidia-smi
```

最终统一生成：

```text
PerformanceReport
```

---

# 23. Experiment Tracking

推荐：

```text
MLflow
```

负责记录：

```text
ExperimentID
DatasetID
QECID
NoiseID

model config
git commit

training metrics
LER
latency

checkpoint
ONNX
TensorRT engine
plots
reports
```

职责划分：

```text
DatasetRegistry
    = data truth

MLflow
    = experiment truth
```

---

# 24. Resource / Job Scheduling

你的场景：

```text
one large server

many CPU cores
multiple NVIDIA GPUs
large RAM
NVMe
```

第一阶段：

```text
LocalExecutor
multiprocessing
subprocess
```

第二阶段任务很多后：

```text
Ray Core
```

定义：

```text
ResourceSpec

cpu = 32
gpu = 2
memory = ...
```

然后：

```text
Experiment
      ↓
ResourceSpec
      ↓
Executor
   /      \
Local      Ray
```

业务层不知道 GPU 是：

```text
cuda:0
cuda:3
```

---

# 25. Environment

推荐：

```text
Python

uv
Git

Docker
NVIDIA Container Toolkit
NGC containers

GitHub Actions
```

HPC 后期：

```text
Apptainer
```

---

# 26. 最终系统主流程

```text
OpenSpec
   ↓

QECSpec
NoiseSpec
DatasetSpec
ExperimentSpec
   ↓

Circuit / Noise Adapter
   ├── Stim
   ├── CUDA-Q
   └── Qiskit
   ↓

Data Generation
   ├── CPU: Stim / Sinter
   ├── GPU: CUDA-Q QEC / cuStabilizer
   └── QPU
   ↓

QECBatch
   ↓

DatasetRegistry
   ├── storage
   └── provenance
   ↓

Data Pipeline
   ├── DataLoader
   ├── DALI
   └── GPU-native
   ↓

PyTorch
   ↓

CNN / GNN / Transformer
   ↓

DDP
Checkpoint
Resume
   ↓

Trained Model
   ↓

DecoderRuntime
   ├── Torch CPU
   ├── Torch CUDA
   ├── ONNX
   ├── TensorRT
   ├── Classical
   ├── Hybrid
   ├── FPGA
   └── ASIC
   ↓

Benchmark
Scientific Validation
Performance Profiling
Golden Verification
   ↓

Realtime QEC / QPU Integration
```

---

# 27. 真正应该自己维护的核心组件

最终真正值得你长期维护的不是几十个基础工具，而是这些：

```text
QECSpec

NoiseSpec

DatasetSpec

QECBatch

DatasetRegistry

ExperimentSpec

DecoderRuntime

VerificationSpec

ResourceSpec

PerformanceReport
```

也就是大约 **10 个核心 contract**。

其余尽可能交给：

```text
Stim
CUDA-Q
cuStabilizer
PyTorch
DDP
DALI
TensorRT
Ray
MLflow
Nsight
pytest
FINN / Vitis
```

来完成。

这就是整个平台最重要的设计原则：

> **自己定义 AI-for-QEC 的语义和实验契约；不要重新实现 AI、GPU、分布式、数据流水线和硬件工具链。**
