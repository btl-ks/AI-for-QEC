# Verification

## Scope

本验证证明 v0.1 技术选择、Registry/配置预检和 adapter/data-pipeline contract 已建立。它不证明 Qiskit、Stim、CUDA-Q、PyTorch 或 PyMatching adapter 已实现，也不证明真实 pinned-memory、non-blocking 或 zero-copy 数据传输已经发生。

## Evidence

### Technology catalog

`openspec/technology-catalog.yaml` 与 `openspec/technology-catalog.schema.json` 通过 YAML/JSON Schema 验证。目录包含 Qiskit circuit、Stim CPU/noise/generator、CUDA-Q GPU/generator、PyTorch CUDA trainer、PyMatching CPU decoder、PyTorch GPU decoder、Tensor-first QECBatch 以及两条数据路径。

所有第三方条目的 implementation 均为 `contracts-only` 或 `none`；DALI、Ray Data 和 custom CUDA extension 为 deferred。

### Registry 与 fail-fast

单元测试覆盖：

- decorator registration 与 `build(key, **kwargs)`；
- 重复注册在覆盖前失败；
- 未知 key 报告 Registry 名称与可选项；
- 13 个完整配置字段路径集中且唯一；
- mapping/sequence 中 `unresolved` 完整路径检测；
- 精确 allow set；
- 未知选项在工厂调用前失败；
- 全局生产 Registry 没有 Stim/PyMatching 假工厂。

### Contract tests

命令：

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q ai_qec tests
```

结果：22 个测试全部通过，compileall 通过。公共 facade 导入不会加载 qiskit、stim、cudaq、torch 或 pymatching。

### OpenSpec

命令：

```bash
openspec validate --all --strict --no-interactive
```

结果：active change 与 7 个既有正式 specs 全部通过。

### Package build

命令：

```bash
uv build --offline
```

结果：sdist 与 wheel 构建成功。wheel 包含 registry、config validation、technology、circuit 和 data pipeline modules；在仓库外隔离虚拟环境安装后，101 个公共名称可导入，且未加载选定第三方框架。

## Capability Decision

- `governance/technology-selection` 与 `governance/configuration-registry`：归档并完成最终 strict validation 后可标记 `validated`。
- `qec/dataset-pipeline`、`qec/decoder-workbench`、`research/training-contracts`：保持 `planned/contracts-only`，因为只有选择与接口，没有第三方执行实现。
