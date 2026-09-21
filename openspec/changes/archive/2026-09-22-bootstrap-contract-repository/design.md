# Design

## Context

当前目录没有 Python 包、测试、正式规格或可验证的实现基线，只有三份架构背景文档。需求要求本次只建立仓库和接口，不实现任何科研或执行算法；同时，接口骨架不得被误认为 capability 已经可用。

## Goals / Non-Goals

**Goals:**

- 让公共契约可被 Python 导入、静态检查和最小构造测试验证。
- 让领域边界与 Golden Path 一致，并为后续有界 change 提供稳定落点。
- 让 OpenSpec strict validation、capability 状态和实现证据保持一致。

**Non-Goals:**

- 不生成 syndrome，不计算 DatasetKey，不读写 registry。
- 不执行训练、解码、checkpoint 恢复或 Accuracy Gate 判定。
- 不创建尚未进入近期范围的 TensorRT、FPGA、ASIC、Ray 或实时 QPU 空目录。
- 不声明论文复现、数据流水线或科学评估已经可用。

## Decisions

### 1. 使用标准库定义契约

值对象使用 frozen、slots dataclass，行为边界使用 `typing.Protocol`，枚举使用 `enum.StrEnum`。这样接口层没有运行时第三方依赖，也不会把 Pydantic、PyTorch 或存储库提前固化为领域 contract。后续实现可以在边界上提供适配器。

备选方案是立即使用 Pydantic；暂不采用，因为序列化版本和验证策略尚未在独立 change 中确定。

### 2. 按领域组织最小文件集

近期只建立 `qec`、`data`、`experiment`、`training`、`models/decoders` 和 `evaluation/scientific`。每个文件要么定义稳定值对象，要么定义实现方必须满足的 Protocol；不使用抛出 `NotImplementedError` 的假入口。

### 3. 公共入口采用 facade

Notebook 和外部调用方从 `ai_qec.notebook_api` 导入公共契约。内部模块仍可直接导入，但兼容性承诺只覆盖 facade 导出的名称。由于本次没有生命周期实现，facade 不提供虚假的 `create_experiment()` 函数，只导出 `ExperimentFactory` Protocol。

### 4. 容器和后端保持泛型

QECBatch 与 DecodeRequest 对底层数组类型使用泛型参数，接口不绑定 NumPy、PyTorch 或 CUDA。Artifact 使用 URI 字符串而非本地 `Path`，允许后续本地或远端存储实现。

### 5. 规格状态与代码状态分离

OpenSpec 规格描述目标行为，但 `capabilities.yaml` 将领域 capability 保持为 `planned` 并注明 `contracts-only`。只有未来 change 提供实现、测试和 verification evidence 后才能升级状态。

### 6. 最小测试只验证契约

标准库 `unittest` 验证公共导入、不可变值对象、枚举和 Protocol 的形状，不测试尚不存在的算法。测试成功仅证明 scaffold 一致，不证明科研 capability 可用。

## Risks / Trade-offs

- [接口过早冻结] → 仅公开 Golden Path 所需最小字段，并允许通过后续 OpenSpec change 演进。
- [dataclass 不能完成配置验证] → 当前只承诺类型结构；运行时验证和序列化由后续 change 决定。
- [泛型数组降低运行时约束] → 后续 adapter 必须通过 schema/version 和专门验证器收紧边界。
- [规格看似已经实现] → capability 清单保留 `planned`，README 与验证文档重复声明 contracts-only。
- [原架构文档与正式规格漂移] → AGENTS 明确 OpenSpec 优先级，后续变更只修改正式规格。

## Migration Plan

1. 创建仓库与 OpenSpec change。
2. 增加接口包、配置样例和契约测试。
3. 运行导入、单元测试和 OpenSpec strict validation。
4. 写入 verification evidence 并归档 bootstrap change，使 delta 成为正式 specs。
5. 后续 capability 每项通过独立 change 实现，不在本次填充算法。
