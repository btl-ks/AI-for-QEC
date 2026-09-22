# Spec Delta

## ADDED Requirements

### Requirement: 配置网格只覆盖已声明字段
配置网格展开 SHALL 从一个基础配置按各轴取值的笛卡尔积生成相互独立的配置，并 SHALL 支持所有点共享的覆盖与按某个轴取值耦合的覆盖。每个覆盖 MUST 指向基础配置中已存在的完整字段路径；同一字段被多个来源覆盖、耦合覆盖缺少某个轴取值，或生成的实验名重复时，展开 MUST 在返回任何配置前失败。展开 MUST NOT 修改基础配置。

#### Scenario: 覆盖路径拼写错误
- **WHEN** 网格覆盖 `training.epoch`，而基础配置中只有 `training.epochs`
- **THEN** 展开 SHALL 报告完整路径 `training.epoch` 并失败，且不返回任何配置

#### Scenario: 展开论文网格
- **WHEN** 以 L ∈ {4, 6} 与 11 个 p 取值展开，并按 L 耦合隐藏单元数与 epoch 数
- **THEN** 展开 SHALL 返回 22 个实验名互不相同的配置，每个配置只在声明的字段上与基础配置不同
