# Verification

## Scope

本验证证明论文 Notebook 源码有效、高层编排顺序可测试、公共 Protocol 可打包导入。它不证明数据生成、训练、解码、恢复、科学评估、Accuracy Gate 或性能评估已经实现。

## Evidence

### Notebook JSON 与 code cells

命令：

```bash
python3 -m json.tool paper/srcs/ai_for_qec_workflow.ipynb
python3 -m unittest tests.unit.test_paper_notebook -v
```

结果：Notebook 是有效 JSON；全部 code cells 可编译并在共享 namespace 中执行；默认 `RUNTIME is None`，没有启动实验。

### 编排顺序与 Gate 行为

记录型 runtime 测试观察到以下顺序：

```text
create_experiment
start_or_recover
resolve_dataset
train
evaluate_accuracy
check_accuracy_gate
visualize
finish
```

Gate FAIL 场景未调用 `evaluate_performance`，并正常执行 `visualize` 与 `finish`。

### Contract 与回归测试

命令：

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q ai_qec tests
```

结果：9 个测试全部通过，compileall 通过。

### Capability registry

`openspec/capabilities.yaml` 通过 `capabilities.schema.json` 验证，共 7 项；`research/paper-reproduction` 为 `planned/contracts-only`。

### OpenSpec strict validation

命令：

```bash
openspec validate --all --strict --no-interactive
```

结果：active change 与 6 个既有正式 specs 全部通过。

### Package build

命令：

```bash
uv build
```

结果：sdist 与 wheel 构建成功。wheel 包含 `ai_qec/paper/`，在仓库外的隔离虚拟环境安装后，可从 `ai_qec.notebook_api` 导入三个 Notebook Protocol。

## Capability Decision

`research/paper-reproduction` 保持 `planned/contracts-only`；Notebook 源码和 Protocol 不是论文复现实现或科学证据。
