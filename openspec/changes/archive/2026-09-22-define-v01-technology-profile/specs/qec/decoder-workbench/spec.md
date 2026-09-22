# Spec Delta

## ADDED Requirements

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

