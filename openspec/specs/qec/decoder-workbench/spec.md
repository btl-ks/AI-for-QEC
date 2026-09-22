# qec/decoder-workbench Specification

## Purpose
为 AI、经典和未来硬件解码器定义统一输入输出边界，使科学评估能够在同一批样本上替换解码实现并保留失败与溯源信息。

## Requirements

### Requirement: 解码器使用统一请求与结果
所有解码器 SHALL 接受包含 QECBatch、解码器身份和上下文的 DecodeRequest，并返回包含预测、失败信息和 provenance 的 DecodeResult。

#### Scenario: 更换解码器实现
- **WHEN** 调用方将 AI decoder 替换为 classical decoder
- **THEN** 调用方 SHALL 无需改变数据 contract 或结果读取方式

### Requirement: 解码失败不得伪装为预测
解码器无法处理输入、超时或未收敛时 SHALL 显式返回失败状态或抛出声明的 contract error，不得返回占位预测或静默使用另一个 decoder。

#### Scenario: 请求不受支持
- **WHEN** decoder 不支持请求的 QECBatch schema
- **THEN** decoder MUST 明确拒绝请求并不得生成看似成功的 DecodeResult

### Requirement: v0.1 CPU 与 GPU decoder 选择
v0.1 SHALL 选择 PyMatching 作为 CPU classical decoder adapter，并选择 PyTorch 作为 GPU AI decoder runtime；二者 SHALL 满足同一 DecodeRequest/DecodeResult contract。

#### Scenario: 同批样本比较 CPU 与 GPU decoder
- **WHEN** 科学评估比较 PyMatching CPU baseline 与 PyTorch GPU decoder
- **THEN** 两者 SHALL 使用相同 QECBatch sample identities、observable truth 和失败语义

### Requirement: Decoder runtime 必须显式
每个 DecodeResult SHALL 记录实际 decoder technology、device 和版本；请求的 decoder 不可用时 MUST NOT 静默改用另一个 runtime。

#### Scenario: PyMatching 不可用
- **WHEN** 评估要求 `pymatching-cpu` 但 adapter 未安装或不兼容
- **THEN** 评估 SHALL 在产生比较结果前失败并明确报告该 decoder unavailable

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
