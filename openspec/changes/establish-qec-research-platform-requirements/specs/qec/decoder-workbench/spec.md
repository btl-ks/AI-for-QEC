# Spec Delta

## Purpose

定义经典、生成式、混合和直接神经解码器共享的输入输出与评估合同，使不同方法能够在相同物理数据、逻辑真值和统计协议下公平比较。

## ADDED Requirements

### Requirement: 解码器使用统一请求与结果语义
系统 SHALL 向 decoder 提供已验证的问题身份、detector events 和声明的 side information，并返回 predicted observable flips 以及适用的恢复、残余 syndrome、收敛和超时诊断。

#### Scenario: Decoder 不支持输入能力
- **WHEN** 数据包含 decoder 未声明支持的 soft readout、leakage 或上下文要求
- **THEN** 系统在解码前拒绝该组合并报告能力不匹配

#### Scenario: Decoder 超时
- **WHEN** 单个或批量样本达到预登记的解码上限
- **THEN** 结果显式记录 timeout 和有效性状态，不得将其静默计为成功预测

### Requirement: 逻辑错误率使用 observable residual 定义
电路级 benchmark SHALL 依据 decoder 预测与 sampled observable flips 的 residual 是否非零计算逻辑失败，并同时报告失败数、shots、LER 和置信区间。

#### Scenario: 多个 decoder 公平比较
- **WHEN** benchmark 比较参考 decoder 与学习型 decoder
- **THEN** 所有方法消费同一 dataset 身份和 shot 集，并使用同一 observable truth、停止规则和 LER 定义

### Requirement: 模型与训练配置真实生效
系统 SHALL 通过能力注册和解析后的配置构建 decoder 与 trainer，并验证每个已声明参数被消费或被明确拒绝。

#### Scenario: 修改有效训练参数
- **WHEN** 用户修改 epochs、batch size、optimizer、scheduler 或 checkpoint 选择规则
- **THEN** resolved plan 和实际训练行为反映该修改

#### Scenario: 模型身份与 checkpoint 不一致
- **WHEN** evaluator 加载的 checkpoint 与模型实现、输入表示、dataset 或 schema 身份不兼容
- **THEN** 系统拒绝评估并指出冲突字段

### Requirement: 参考基线与学习方法同表报告
系统 SHALL 至少支持一个可扩展经典解码基线，并在相同协议中报告经典、RBM、混合和直接神经方法中实际已实现的项目。

#### Scenario: 某类方法尚未实现
- **WHEN** 对照表请求尚未注册的方法
- **THEN** 系统标记该能力不可用并阻止把空值或占位结果解释为实验结果

### Requirement: 解码输出可校准和诊断
对概率型 decoder，系统 SHALL 报告与其输出语义一致的 proper loss、校准指标和分层结果；对迭代型 decoder，系统 SHALL 报告步数、收敛与失败原因。

#### Scenario: 概率输出参与层级决策
- **WHEN** decoder 的置信度用于拒绝、路由或后续经典解码
- **THEN** benchmark 同时报告校准、覆盖率、逻辑错误率和路由成本

### Requirement: 研究入口共享同一实现
脚本、Notebook 和批量 runner SHALL 复用同一公开解码、训练和评估接口，且新增公开 API MUST 通过项目统一入口暴露。

#### Scenario: Notebook 执行论文实验
- **WHEN** 用户从清空的 kernel 顺序执行 Notebook
- **THEN** Notebook 只编排共享领域接口并生成可追溯 run，不复制训练或解码核心循环
