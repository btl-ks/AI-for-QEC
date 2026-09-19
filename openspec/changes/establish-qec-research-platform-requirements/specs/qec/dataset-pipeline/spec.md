# Spec Delta

## Purpose

定义 QEC 样本从电路和噪声规格到不可变数据集的统一行为，使训练、测试和不同采样后端共享可验证的 detector、observable、划分与来源语义。

## ADDED Requirements

### Requirement: 数据保留 QEC 真值与来源
数据集 SHALL 保存非空的 detector events、observable flips、形状与位序、code/circuit/noise 身份、生成后端、版本、seed、schema 和内容哈希。

#### Scenario: 生成有效电路级数据
- **WHEN** sampler 完成一批 QEC 电路样本
- **THEN** writer 校验必需字段、二值范围、shape、dtype、样本数和来源后才提交 dataset

#### Scenario: 生成结果为空或损坏
- **WHEN** sampler 返回零样本、缺失 observable truth 或不符合 schema 的数组
- **THEN** 系统失败且不得写出可被后续流程接受的占位数据集

### Requirement: 数据可流式生成与加载
系统 SHALL 支持按 shard 和 batch 生成、校验与加载数据，而无需将正式规模的数据集全部驻留内存。

#### Scenario: 生成大规模样本
- **WHEN** 样本规模超过单个 shard 的配置上限
- **THEN** 系统创建多个带顺序、样本范围和独立哈希的 shard，并由 manifest 汇总总样本数

#### Scenario: 消费最后一个不完整批次
- **WHEN** split 的样本数不能被 batch size 整除
- **THEN** loader 返回真实大小的最后批次且指标按实际样本数聚合

### Requirement: Split 防止信息泄漏
系统 SHALL 按实验定义的最高泄漏层级隔离 train、validation 和 test，并记录 group、session、domain 或 pair 的划分审计。

#### Scenario: Counterfactual pair 参与划分
- **WHEN** 两个样本共享同一 matched pair 或上层实验来源
- **THEN** split 策略按照预登记规则将相关样本保持在允许的同一分组中且报告交叉检查

#### Scenario: 检测到跨 split 泄漏
- **WHEN** audit 发现受约束的 group、session、domain 或 pair 同时出现在互斥 split
- **THEN** 系统拒绝正式训练或评估

### Requirement: 采样后端必须显式选择
系统 SHALL 显式记录请求和解析后的 sampler backend，并在执行前检查依赖、设备与目标能力；系统 MUST 不得自动切换后端。

#### Scenario: GPU backend 不可用
- **WHEN** 配置请求 GPU sampler 但依赖、设备或目标功能检查失败
- **THEN** 系统失败且不创建 dataset，也不回退到 CPU sampler

#### Scenario: 新后端通过兼容验证
- **WHEN** 新 sampler backend 在相同问题规格下通过 schema、物理约束和预登记统计等价性测试
- **THEN** 系统允许其通过统一 sampler 契约生成独立标明后端身份的数据集

### Requirement: 生成结果可复现
系统 SHALL 定义 seed、分片和批次对样本流的影响，并保证相同生成规格可重建相同内容，或把无法消除的批次因素纳入身份。

#### Scenario: 改变内部 batch 大小
- **WHEN** backend 声明样本流与 batch 分块无关
- **THEN** 相同生成规格与 seed 在不同 batch 大小下产生相同内容哈希

### Requirement: 训练数据供应不得改变测试真值
预取、缓存、困难样本选择和设备传输 SHALL 只改变训练样本的供给顺序或权重，不得修改独立测试集及其 observable truth。

#### Scenario: 开启困难样本训练
- **WHEN** trainer 使用在线候选池和困难样本缓存
- **THEN** test split 保持固定、只读且不参与样本评分器更新
