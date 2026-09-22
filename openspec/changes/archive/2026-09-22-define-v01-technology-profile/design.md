# Design

## Context

现有 contract 使用泛型数组和字符串 device，能够保持 vendor-neutral，但无法回答 v0.1 实际优先实现哪个 adapter，也无法表达 host/device residency 与 transfer policy。用户已经给出明确技术矩阵，因此本 change 将“领域语义”和“选定实现”分层记录。

## Goals / Non-Goals

**Goals:**

- 建立唯一的机器可读 v0.1 technology profile。
- 用无第三方依赖的值对象和 Protocol 描述 circuit、noise compilation、batch representation 及两条数据路径。
- 用统一 Registry 和配置预检替代散落的硬编码技术分支。
- 让后续 adapter change 可以逐项实现并独立升级状态。

**Non-Goals:**

- 不安装或导入 Qiskit、Stim、CUDA-Q、PyTorch、PyMatching。
- 不实现 tensor conversion、DLPack exchange、DataLoader、CUDA stream 或 decoder。
- 不在 v0.1 引入 DALI、Ray Data 或自定义 CUDA extension。
- 不承诺真实 zero-copy；只定义协商、验证和证据边界。

## Decisions

### 1. Vendor-neutral core，selected adapter profile

核心 dataclass 不直接持有第三方类型。`QECCircuitAdapter[CircuitT]`、`NoiseCompiler[NoiseT]`、`QECDataGenerator[ArrayT]` 和 transfer Protocol 使用泛型承载第三方对象；technology ID 使用稳定 kebab-case 字符串。

Qiskit circuit 与 Stim noise 不会成为 QECSpec/NoiseSpec 的字段类型。它们是 v0.1 catalog 中选定的 adapter，避免未来更换工具时破坏数据与实验身份 contract。

### 2. Tensor-first 不等于 PyTorch-only storage

QECBatch 增加 `BatchLayout`，记录 representation、device、dtype、shape、memory residency、interchange 和 zero-copy claim。训练/推理主路径选择 PyTorch Tensor；数据集持久化仍允许 bit-packed/NumPy，GPU 交换允许 CUDA Tensor/DLPack。

### 3. 分开定义 H2D 与 D2D pipeline

`CPUToGPUDataPipeline` 以 `CPUToGPUPipelineSpec` 声明 Dataset/DataLoader、pin memory、prefetch 和 non-blocking。`GPUToGPUDataPipeline` 以 `GPUToGPUPipelineSpec` 声明 source/target runtime、interchange、device/stream compatibility、zero-copy preference 与 host-staging policy。

这避免用一个模糊的 `transfer()` 隐藏完全不同的性能与正确性风险。

### 4. Catalog 状态独立于 capability 状态

technology catalog 的 `selection: primary` 表示决策，`implementation: contracts-only` 表示尚无代码。治理 capability 可在 catalog/schema/validation 完成后标记 validated；QEC、trainer、decoder capability 仍保持 planned。

### 5. 完整字段路径驱动的 Registry

通用 `Registry[T]` 接受 class 或 function factory，提供 decorator registration、重复检测、key 查询和 `build(key, **kwargs)`。全局注册表名称直接使用配置完整路径，例如 `qec.code_family`、`noise.family`、`dataset.generator`、`training.optimizer`，避免不同 section 都使用含义模糊的 `family`。

`REGISTRIES_BY_PATH` 是唯一字段绑定表。组装代码通过 `build_from_config(config, path, **kwargs)` 读取 key 并调用 Registry，不写 `if technology == ...`。选定技术在 adapter 真正实现前不注册，从而使“selected but unavailable”自然 fail fast。

### 6. 两层构造前预检

第一层 `validate_no_unresolved` 递归检查 Mapping 与 Sequence，并支持精确路径 allow set，适合分阶段只构建已确定 section。第二层 `validate_registered_selections` 检查配置中出现的注册字段是否已登记；未知选项不 fallback。

配置预检不负责 schema 类型验证，后续可在 Pydantic/config schema change 中补充。当前它专注于未决占位值与实现可用性。

## Risks / Trade-offs

- [CUDA-Q 与 PyTorch 的真实零拷贝能力依版本而变] → contract 要求显式协商与证据，不预先声称 zero-copy。
- [Qiskit circuit 到 Stim/CUDA-Q 的语义损失] → adapter 必须产生兼容性结果；不支持时拒绝或显式 approximation。
- [Tensor-first 使 CPU 持久化细节复杂] → BatchLayout 将 persistent representation 与 runtime representation 分开记录。
- [具体工具版本尚未锁定] → catalog 先使用版本约束 `unresolved`；实现 change 必须锁定并验证版本后才能升级。
- [全局 Registry 被测试或插件污染] → Registry 支持独立实例；全局项只由显式 adapter import 注册，测试优先使用局部 Registry。

## Migration Plan

1. 新增 technology catalog/schema、Registry、配置预检和接口类型。
2. 扩展 Notebook facade 与契约测试。
3. 严格验证 OpenSpec、catalog 和 wheel。
4. 归档后将 technology-selection 与 configuration-registry 治理 capability 标记 validated。
5. 后续按 Qiskit、Stim CPU、CUDA-Q GPU、PyTorch trainer、PyMatching decoder 分拆实现 changes，并只注册真实可用工厂。
