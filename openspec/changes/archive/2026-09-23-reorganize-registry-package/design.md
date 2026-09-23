# Design

## Context

动机见 `proposal.md`。当前四个顶层模块形成单向依赖链：`registry.py` 提供无依赖核心，`registries.py` 声明全局实例，`config_validation.py` 消费二者，`implementations.py` 通过显式导入填充全局实例。`ai_qec.notebook_api` 公开其中的稳定符号，但不得加载可选 runtime。项目同时要求使用无 `__init__.py` 的隐式 namespace，并从定义符号的叶子模块导入。

## Goals / Non-Goals

**Goals:**

- 使 Registry 子系统的物理目录与依赖层次一致。
- 保持 `ai_qec.notebook_api.__all__`、Registry key、异常类型、构造和校验语义不变。
- 保持 `notebook_api` 导入零注册且不加载可选 runtime。

**Non-Goals:**

- 不保留旧的非公共模块路径兼容层。
- 不改变技术目录、配置 schema、实现发现机制或已注册工厂集合。
- 不新增 GPU、FPGA、ASIC、QPU 或分布式编排能力。

## Decisions

### 使用 `ai_qec.registry` namespace，而非 `ai_qec.register`

`Registry` 是领域中的名词和现有核心抽象；`register` 更适合作为方法名。新目录不创建 `__init__.py`，符合现有隐式 namespace 打包约束。

备选方案是使用 `ai_qec.configuration`，但该名称范围过宽，容易把实验配置模型、网格展开和技术目录等不同职责继续聚合到一起。

### 按依赖层拆为四个明确叶子模块

- `ai_qec.registry.core`：`Registry`、错误类型和工厂类型。
- `ai_qec.registry.catalog`：全局 Registry 实例和 `REGISTRIES_BY_PATH`。
- `ai_qec.registry.validation`：`unresolved`、字段路径、注册选择与构造前校验。
- `ai_qec.registry.bootstrap`：延迟导入内置可执行实现。

依赖方向固定为 `core ← catalog ← validation/bootstrap`；`bootstrap` 不由公共 facade 导入。备选方案是保留原文件名放入目录，但 `registries` 与 `implementations` 的复数命名无法准确表达“集中目录”和“加载入口”的职责。

### 公共兼容性只通过既有 facade 维持

`ai_qec.notebook_api` 改为从新叶子模块导入，并维持完全相同的 `__all__`。旧的 `ai_qec.registry` 模块路径本来不是公共入口，迁移后不增加转发模块或 `__init__.py` 聚合导出，以满足明确叶子模块导入规则。

### 以导入与打包测试约束迁移完整性

除现有 Registry 行为测试外，打包测试将覆盖四个新叶子模块均可导入、旧顶层文件消失、源码和测试树仍无 `__init__.py`。现有公共导出摘要与子进程延迟加载测试继续作为不变性证据。

## Risks / Trade-offs

- [遗漏内部导入导致运行时错误] → 全仓搜索旧模块路径，并运行完整测试与严格类型/风格检查。
- [迁移意外触发 implementation 导入] → `notebook_api` 只导入 `core`、`catalog` 和 `validation`，继续用独立子进程检查注册数量及可选依赖加载状态。
- [旧内部路径的下游调用方受影响] → 项目只承诺 `ai_qec.notebook_api` 公共入口；README 和仓库内调用方同步迁移，不添加违反叶子导入约束的兼容层。
- [namespace 未进入 wheel] → 通过 wheel 构建和仓库外安装检查确认 `ai_qec/registry/*.py` 被发现。

## Migration Plan

1. 建立新目录并迁移四个模块内容。
2. 原子更新源码、测试和 README 中的导入路径，再删除旧顶层模块。
3. 运行定向测试、完整测试、静态检查、OpenSpec 严格验证和 wheel 检查。
4. 若验证失败，回退本 change 的模块移动和导入更新；不修改已有数据或运行记录。
