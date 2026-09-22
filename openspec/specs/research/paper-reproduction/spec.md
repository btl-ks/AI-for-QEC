# research/paper-reproduction Specification

## Purpose
定义论文 Notebook 源码的保存位置、公共编排边界和可验证执行顺序，使论文工作流能够被版本控制与审查，同时不会把尚未实现的研究 runtime 伪装为可用能力。

## Requirements

### Requirement: Notebook 源码纳入版本控制
论文实验 Notebook 源码 SHALL 保存于 `paper/srcs/`，文件 MUST 是有效的 Jupyter Notebook JSON，并 SHALL 包含论文或工作流身份、配置、编排函数和状态说明。

#### Scenario: 打开论文源码
- **WHEN** 维护者或审查者读取 `paper/srcs/` 下的 `.ipynb`
- **THEN** Notebook SHALL 可被标准 Jupyter 工具解析，且配置 cell SHALL 是合法 Python

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

### Requirement: Notebook 定义 cells 无副作用
Notebook 中定义配置、编排函数和辅助函数的 cells SHALL 在导入或执行时不创建 runtime、Run、dataset、model、metric 或 performance Artifact；启动实验的 cells SHALL 用 cell 标签 `run-experiment` 显式标记，并 SHALL 只通过显式构造的 runtime 调用 `run_paper`。

#### Scenario: 只执行定义 cells
- **WHEN** 从干净内核执行全部未标记 `run-experiment` 的 code cells
- **THEN** 执行 SHALL 成功完成，SHALL 定义 `CONFIG` 与 `run_paper`，且 SHALL NOT 创建 runtime 或任何 Artifact

#### Scenario: Run All
- **WHEN** 用户在安装运行时依赖的内核中执行全部 cells
- **THEN** 标记 `run-experiment` 的 cells SHALL 构造本地 runtime 并按声明网格执行 `run_paper`

### Requirement: Torlai–Melko 2017 复现协议
论文 Notebook SHALL 复现 Torlai & Melko (PRL 119, 030501, 2017) 的 toric code 相位翻转实验：对声明网格中的每个 (L, p) 点独立训练联合 RBM，以算法 1 解码 test split，并在同一 test split 上与 MWPM 比较；Notebook SHALL 展示 P_fail–p 曲线（图 3）与 RBM 残余同调类直方图（图 4），并 SHALL 标明所用超参数、样本数和与论文设置的差异。

#### Scenario: 执行论文网格
- **WHEN** Notebook 以 `paper` profile 执行
- **THEN** 每个 (L, p) 点 SHALL 拥有独立 Experiment、ScientificEvaluationResult 与 ScientificAcceptanceResult，且图 3 SHALL 同时显示两个 decoder 的点估计与置信区间

#### Scenario: 冒烟 profile
- **WHEN** Notebook 以 `smoke` profile 执行
- **THEN** Notebook SHALL 使用小规模配置走完同一编排路径，并 SHALL 明确说明其结果不能作为科学结论
