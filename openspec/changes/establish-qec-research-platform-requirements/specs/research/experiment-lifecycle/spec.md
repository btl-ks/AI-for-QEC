# Spec Delta

## Purpose

定义科研实验从配置解析、数据引用、执行到导出结束的可追溯行为，使每项结果都能绑定真实代码、环境、数据和终态。

## ADDED Requirements

### Requirement: 执行前解析完整计划
系统 SHALL 在创建 dataset 或 run 之前解析完整配置、默认值、能力依赖、步骤顺序和输出契约，并拒绝未知、未消费或当前不可用的选项。

#### Scenario: 配置引用未实现能力
- **WHEN** 正式配置请求尚未注册或依赖不可用的 sampler、decoder、trainer 或 deployment backend
- **THEN** 系统在写入 dataset 或 run 前失败，并明确报告缺失能力且不执行静默回退

#### Scenario: 执行计划有效
- **WHEN** 配置字段、能力和步骤依赖均有效
- **THEN** 系统生成包含有效参数、步骤、输出契约和身份摘要的 resolved plan

### Requirement: Dataset 与 run 身份分离
系统 SHALL 仅使用影响数据内容的生成规格计算 dataset 身份，并为每次执行分配独立 run 身份。

#### Scenario: 仅修改训练参数
- **WHEN** 用户保持 code、circuit、noise、seed 和采样规格不变，仅修改模型或训练参数
- **THEN** 系统复用通过完整性校验的同一 dataset，并创建新的 run

#### Scenario: 修改生成规格
- **WHEN** 用户修改任一影响样本内容的生成字段
- **THEN** 系统产生不同 dataset 身份且不得把旧数据视为匹配

### Requirement: 产物不可变且可验证
系统 SHALL 为 dataset、checkpoint、预测、指标和导出包记录 schema、内容哈希及其依赖身份；已登记产物 MUST 不被后续阶段改写。

#### Scenario: 后续阶段需要汇总结果
- **WHEN** benchmark 需要组合已有评估指标
- **THEN** 系统写入新的汇总产物并保持原评估产物及其哈希不变

#### Scenario: 产物被外部修改
- **WHEN** run 结束校验发现已登记产物内容与记录哈希不一致
- **THEN** 系统将 run 判定为失败并指出不一致的产物

### Requirement: Run 具有明确终态
系统 SHALL 记录 run 的进程身份、阶段时间线和 success、partial、failed 或 interrupted 终态，且不得将零步骤或部分执行标记为 success。

#### Scenario: 零步骤或部分执行
- **WHEN** run 没有执行任何步骤，或只完成了部分步骤
- **THEN** 系统记录 partial 或 failed 终态，且不得标记为 success

### Requirement: 正式结果可独立复核
系统 SHALL 为正式 run 保存配置快照、环境、代码状态、数据引用、日志、指标和引用关系，并通过显式 allowlist 导出最小复现包。

#### Scenario: 导出论文复现包
- **WHEN** 用户导出一个已完成且身份校验通过的正式 run
- **THEN** 导出包包含重算声明指标所需的最小文件、锁定依赖和验证入口，且不包含未授权项目内容
