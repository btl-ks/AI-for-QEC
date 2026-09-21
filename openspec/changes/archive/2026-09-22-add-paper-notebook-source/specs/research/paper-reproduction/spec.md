# Spec Delta

## Purpose

定义论文 Notebook 源码的保存位置、公共编排边界和可验证执行顺序，使论文工作流能够被版本控制与审查，同时不会把尚未实现的研究 runtime 伪装为可用能力。

## ADDED Requirements

### Requirement: Notebook 源码纳入版本控制
论文实验 Notebook 源码 SHALL 保存于 `paper/srcs/`，文件 MUST 是有效的 Jupyter Notebook JSON，并 SHALL 包含论文或工作流身份、配置、编排函数和状态说明。

#### Scenario: 打开论文源码
- **WHEN** 维护者或审查者读取 `paper/srcs/` 下的 `.ipynb`
- **THEN** Notebook SHALL 可被标准 Jupyter 工具解析，且配置 cell SHALL 是合法 Python

### Requirement: contracts-only Notebook 可安全执行
当仓库尚无 runtime 实现时，Notebook SHALL 只定义配置和接受显式 runtime 参数的编排函数，MUST NOT 在导入或 clean `Run All` 时调用不存在的执行实现。

#### Scenario: 在接口仓库中 Run All
- **WHEN** Notebook 在当前 contracts-only 仓库中从干净内核执行全部 cells
- **THEN** 执行 SHALL 成功完成且 SHALL NOT 生成 dataset、model、metric 或 performance Artifact

### Requirement: Notebook 保持标准研究顺序
编排接口 SHALL 按 Configure、Start/Recover、Resolve Dataset、Train、Scientific Evaluation、Accuracy Gate、条件式 Performance Evaluation、Visualize、Finish 的顺序表达论文工作流。

#### Scenario: Accuracy Gate 未通过
- **WHEN** 科学评估完成但 Accuracy Gate 返回 FAIL
- **THEN** Notebook SHALL 跳过 performance evaluation，并仍执行可视化与 Run 收尾

### Requirement: Notebook 只依赖公共 contract
Notebook SHALL 通过 `ai_qec.notebook_api` 使用公共项目 contract；核心数据生成、训练、解码、Artifact 管理和生命周期行为 MUST NOT 复制到 Notebook cells 中。

#### Scenario: 审查 Notebook 代码
- **WHEN** 维护者检查 Notebook 编排源码
- **THEN** Notebook SHALL 只包含配置、小型编排 glue、说明与展示入口，不得包含后端算法实现
