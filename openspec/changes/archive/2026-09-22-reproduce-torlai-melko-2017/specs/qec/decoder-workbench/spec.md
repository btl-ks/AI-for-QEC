# Spec Delta

## ADDED Requirements

### Requirement: Syndrome 钳制 Gibbs 解码
RBM 神经解码器 SHALL 按论文算法 1 执行：syndrome 层钳制为测得 syndrome，误差层与隐藏层随机初始化并交替块 Gibbs 采样；burn-in 之后第一个满足 `S(e) = S₀` 的误差链 SHALL 被选为恢复链。步数预算耗尽仍未找到兼容链的样本 MUST 被报告为超时样本，MUST NOT 获得看似有效的预测。

#### Scenario: 找到兼容链
- **WHEN** 某样本在 burn-in 之后的第 t 步首次采样到 syndrome 等于 `S₀` 的误差链
- **THEN** 解码器 SHALL 以该链计算逻辑可观测量预测，且该恢复链的 syndrome MUST 等于 `S₀`

#### Scenario: 步数预算耗尽
- **WHEN** 某样本在 `max_steps` 内没有采样到兼容链
- **THEN** DecodeResult SHALL 为 `timed-out` 状态，把该样本列入 `failed_sample_ids`，且该样本的预测 MUST 标记为无效值而不是 0/1 位

### Requirement: MWPM baseline 使用均匀边权
PyMatching baseline SHALL 在与 AI decoder 相同的 QECBatch 上，以码的奇偶校验矩阵和均匀边权（论文中的曼哈顿距离）求最小权重完美匹配，并输出逻辑可观测量预测；输入宽度与码不匹配时 MUST 返回 `unsupported` 且不提供预测。

#### Scenario: 比较 AI 与 MWPM
- **WHEN** 科学评估对同一 test split 运行 RBM decoder 与 PyMatching baseline
- **THEN** 两个 DecodeResult SHALL 覆盖相同 sample identities，并各自记录实际 technology、版本和 device
