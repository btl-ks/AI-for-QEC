# Design

## Context

动机见 proposal.md。Experiment 身份是完整配置的摘要，所以网格展开必须是确定性的：相同输入必须生成逐字节相同的配置。

## Goals / Non-Goals

**Goals:**

- Notebook 只声明参数（基础配置、轴、耦合覆盖、共享覆盖、名称模板），不再手写 deepcopy 与嵌套下标。
- 机制与论文无关，可复用于后续论文 Notebook。

**Non-Goals:**

- 不做类型或取值校验；展开结果仍由 `LocalNotebookPlatform.create_experiment` 做完整预检。

## Decisions

### 1. 轴以别名声明

`axes={"L": ("qec.distance", (4, 6)), "p": ("noise.parameters.physical_error_rate", rates)}`。别名同时用于名称模板（`"…-l{L}-p{p:.2f}"`，即 `str.format`）、耦合覆盖的键和返回的 `GridPoint.values`。直接用点分路径做模板字段不可行，因为 `str.format` 把 `.` 解释为属性访问。

### 2. 覆盖来源互斥

共享覆盖、轴与耦合覆盖作用在同一字段即报错，而不是规定优先级。显式冲突比“后者覆盖前者”更容易审查，也避免同一参数在两处定义而只有一处生效。`experiment.name` 由名称模板设置，同样不可再被覆盖。

### 3. 只允许覆盖已存在的叶子路径

路径必须在基础配置中存在，且目标不是 mapping，这样拼错的字段不会静默成为新键；需要新字段时应先加入基础配置。

### 4. 展开顺序与身份

按 `axes` 的声明顺序做笛卡尔积，第一个轴在最外层；每个点深拷贝基础配置后按“共享覆盖 → 轴 → 耦合覆盖 → 名称”写入。实验名改用模板后，Experiment 身份改变（见 proposal 的 BREAKING 说明）；DatasetKey 不含实验名，因此数据集继续复用。

## Risks / Trade-offs

- [首次 Run All 需重新训练约 35 分钟] → 在归档前于后台完成，保证 Notebook 缓存与本 change 一致。
- [旧实验目录成为孤立记录] → 保留，不删除不可变 Artifact；README 不依赖它们。
