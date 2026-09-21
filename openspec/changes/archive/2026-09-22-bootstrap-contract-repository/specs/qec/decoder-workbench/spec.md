# Spec Delta

## Purpose

为 AI、经典和未来硬件解码器定义统一输入输出边界，使科学评估能够在同一批样本上替换解码实现并保留失败与溯源信息。

## ADDED Requirements

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
