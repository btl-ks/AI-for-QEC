# Spec Delta

## Purpose

定义从离线 profiler 到实时异构部署的性能证据标准，使模型计算、完整 decoder 和端到端反馈链路的时延、吞吐、资源与正确性可以分别验证。

## ADDED Requirements

### Requirement: 性能边界必须显式区分
系统 SHALL 分别定义并报告模型计算时间、包含预处理与后处理的 decoder 时间，以及包含传输和反馈的端到端时间。

#### Scenario: 只测量模型 forward
- **WHEN** benchmark 未包含输入转换、数据移动或输出处理
- **THEN** 结果只能标记为 model latency，不能标记为 decoder 或端到端延迟

### Requirement: 性能报告包含可复核条件
每项性能结果 SHALL 记录硬件、软件版本、设备状态、输入形状、batch、warm-up、同步边界、重复次数、p50/p95/p99、吞吐和内存。

#### Scenario: GPU 延迟测试
- **WHEN** benchmark 在异步 GPU 上执行
- **THEN** 计时协议在正确边界同步设备，并分别报告 batch=1 延迟和批量吞吐

### Requirement: 性能与正确性联合验收
部署 benchmark SHALL 在同一构建和数据协议下同时验证预测正确性、LER 或等价结果以及 deadline miss rate。

#### Scenario: 优化后结果发生变化
- **WHEN** 量化、融合、原生实现或硬件后端改变输出
- **THEN** 系统按预登记容差比较参考实现，并在性能结论旁报告准确率或 LER 变化

### Requirement: 仿真生成与解码计时分离
系统 SHALL 分开报告训练/测试数据生成、主机到设备传输、解码和结果验证的耗时，并另外提供完整流水线时间。

#### Scenario: 使用预生成测试数据测 decoder
- **WHEN** 目标是比较 decoder 推理延迟
- **THEN** 仿真数据生成不计入 decoder latency，但其来源和预加载状态必须记录

### Requirement: 部署后端遵守稳定协议
CPU、GPU、原生运行时和 FPGA/ASIC adapter SHALL 使用同一逻辑输入输出语义，显式声明支持能力，并在不可用时失败。

#### Scenario: 原生后端缺少 side information 支持
- **WHEN** 请求包含后端未实现的输入模态
- **THEN** 系统拒绝该部署组合且不使用行为不同的替代路径

### Requirement: 硬件证据区分实测与估算
系统 SHALL 将真实设备测量、综合结果、仿真估算和软件投影明确标注，并为 FPGA/ASIC 结果报告适用的时钟、资源、功耗来源和约束。

#### Scenario: 只有 FPGA 综合估算
- **WHEN** 尚未在真实板卡上运行 decoder
- **THEN** 报告标记为综合或仿真结果，不得声称已经实现板级端到端实时解码
