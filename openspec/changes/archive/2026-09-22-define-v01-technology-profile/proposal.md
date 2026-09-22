# Proposal

## Why

平台已有 vendor-neutral contract，但尚未正式记录 v0.1 的具体技术实现选择，也没有表达 CPU→GPU 与 GPU→GPU 数据路径的接口。需要将 Qiskit、Stim、CUDA-Q、PyTorch 和 PyMatching 固化为可审计的技术基线，同时继续明确这些选择目前只有接口、没有 adapter 实现。

## What Changes

- 新增受 schema 约束的 v0.1 technology catalog，记录 primary、alternative、phase 和 implementation status。
- 指定 Qiskit 为 QEC circuit adapter，Stim 为 CPU QEC/noise/syndrome backend，CUDA-Q 为 GPU QEC/syndrome backend。
- 指定 PyTorch/CUDA 为 trainer runtime，PyMatching 为 CPU decoder，PyTorch 为 GPU decoder。
- 指定 PyTorch Tensor 为 QECBatch 主要运行时表示，并保留 NumPy、bit-packed CPU buffer、CUDA Tensor 与 DLPack 边界。
- 定义 CPU→GPU DataLoader/pinned-memory/non-blocking 接口和 GPU→GPU CUDA-Q/PyTorch zero-copy-preferred 接口。
- 新增统一 Registry 与“配置字段路径 → Registry”声明表，后续 adapter 使用装饰器注册，组装器不硬编码第三方分支。
- 新增递归 `unresolved` 检测和选项预检，所有构造在产生持久化副作用前 fail fast。
- 明确 v0.1 不引入 DALI、Ray Data 或自定义 CUDA extension 实现。
- 不实现任何第三方 adapter、数据搬运、trainer 或 decoder。

## Capabilities

### New Capabilities

- `governance/technology-selection`: 管理正式技术目录、primary/alternative 关系、阶段和实现状态。
- `governance/configuration-registry`: 管理配置字段命名、实现注册、未知选项和未决占位值的构造前校验。

### Modified Capabilities

- `qec/dataset-pipeline`: 增加选定 circuit/noise/generator 后端、Tensor-first QECBatch 与 CPU/GPU 数据路径要求。
- `qec/decoder-workbench`: 增加 PyMatching CPU 与 PyTorch GPU decoder 的 v0.1 选择要求。
- `research/training-contracts`: 增加 PyTorch/CUDA trainer 与 DataLoader 执行基线要求。

## Impact

- 新增 `openspec/technology-catalog.yaml` 及 JSON Schema。
- 新增通用 Registry、配置预检，以及 circuit、noise compiler、runtime representation 和 data-transfer Protocol/值对象。
- 扩展公共 Notebook facade，但不增加 Qiskit、Stim、CUDA-Q、PyTorch 或 PyMatching 运行时依赖。
- capability 仍保持 `planned/contracts-only`，不会因技术目录或接口出现而升级。
