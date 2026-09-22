# Design

## Context

动机见 proposal.md。现状：

- Gate 只有 `paired-non-inferiority`：用 Newcombe (1998) 方法 10 求 `LER_AI − LER_baseline` 的配对区间，上界 ≤ `tolerance` 时 PASS。
- Experiment 身份是完整配置的摘要；复用只发生在同一 Experiment 的恢复链内（`_reusable` 读取恢复源 Attempt 的 Stage 记录）。
- 数据集已经按 DatasetKey 跨 Experiment 复用，DatasetKey 不含实验名和训练字段。
- `runs/<experiment>/config.json` 保存每个 Experiment 的规范化配置快照，Stage 记录保存输入输出的 ArtifactRef（URI 相对项目根，带 checksum）。

## Goals / Non-Goals

**Goals:**

- 相对容差 Gate 的区间方法与现有绝对容差 Gate 属于同一族（MOVER + Wilson），δ = 0 时与现有区间逐位相同。
- 只改 Gate 时，重跑整个网格不训练任何 epoch，也不重跑科学评估和性能评估。
- 复用对已有的 `runs/` 立即生效，不要求旧记录补写任何字段。

**Non-Goals:**

- 不跨 Experiment 从训练中途恢复。
- 不引入可变的全局复用索引文件，因此也不需要为它加锁。
- 不把代码版本或运行环境纳入复用键；这与现有 Experiment 身份的语义一致。

## Decisions

### 1. 相对容差用线性对比的 MOVER 区间

检验 H0: `p_AI ≥ (1+δ)·p_base` 对 H1: `p_AI < (1+δ)·p_base`，等价于对 `θ = p_AI − r·p_base`（r = 1+δ）检验 `θ ≥ 0`。

MOVER（Zou & Donner 2008）对相关估计量的线性组合给出闭式区间：每个分量用自己的区间，按相关系数组合。`r·p_base` 的 Wilson 区间就是 `(r·l₂, r·u₂)`，与 `p_AI` 的相关系数仍为 φ（缩放不改变相关）。代入 Newcombe 方法 10 的公式：

```text
θ̂ = p₁ − r·p₂
L = θ̂ − sqrt((p₁ − l₁)² − 2φ·r·(p₁ − l₁)(u₂ − p₂) + r²(u₂ − p₂)²)
U = θ̂ + sqrt((u₁ − p₁)² − 2φ·r·(u₁ − p₁)(p₂ − l₂) + r²(p₂ − l₂)²)
```

r = 1 时就是现有实现。因此实现为 `paired_difference_interval(..., scale=1.0)` 增加一个参数，现有调用不变。PASS 条件为 `U ≤ 0`。

考虑过的替代方案：

- **用观测到的 baseline LER 乘 δ 作为绝对容差。** 容差会随数据波动，不是预先登记的量，而且忽略了 baseline 估计的方差。
- **对数比值的 Wald 区间，或 Tang (2003) 的配对 score 检验。** 前者在失败数较小时不稳定；后者要求解约束 MLE，与现有 Newcombe 实现不同族，也更难审查。

缺少可以逐位对照的参考实现，所以用两类证据代替：r = 1 时与现有函数逐位一致；在边界 `p_AI = (1+δ)·p_base` 上做 Monte Carlo 模拟，检查 PASS 频率接近单侧 2.5%。

规则名为 `paired-relative-non-inferiority`，不新增配置键：`comparison_rule` 本来就枚举规则，`tolerance` 的含义由规则决定。现有绝对规则保持允许负 tolerance（那表示要求优效）；相对规则要求 δ ≥ 0，在 `parse_experiment_config` 中校验，所以预检阶段就会失败。

证据文件 `acceptance.json` 增加 `rule_parameters`：`{"relative_tolerance": δ, "scale": r}`，并记录 `ratio_estimate = p₁/p₂`（p₂ = 0 时为 null）。

### 2. Stage 复用键 = Stage 名 + 输入引用 + 删去下游字段的配置

```text
key = sha256_json({"stage": name, "inputs": [ArtifactRef...], "config": config − EXCLUDED[name]})
```

| Stage | 从配置中删去 |
|---|---|
| training | `experiment.name`, `model.decoding`, `scientific_evaluation`, `accuracy_gate`, `performance` |
| scientific_evaluation | `experiment.name`, `accuracy_gate`, `performance` |
| performance | `experiment.name`, `accuracy_gate` |

采用删除表而不是保留表：新增的配置字段默认进入复用键，最坏结果是多算一次，而不会错误复用。删除表中的每一项都能说明理由：

- `experiment.name` 不参与任何计算（随机流只由 `master_seed` 派生）。
- 训练不读 `model.decoding`，也不读评估、Gate 和性能设置。
- 科学评估不读 Gate 和性能设置。
- 性能评估只在 Gate PASS 后运行，但测量本身不读 Gate 设置。它依赖 `scientific_evaluation.baseline_decoders`，所以保留该字段。

输入引用进入键：训练的输入是数据集引用，科学评估和性能评估的输入是数据集加模型引用。复用来的模型引用与来源完全相同，所以下游 Stage 的键能继续匹配。

Accuracy Gate 与可视化不跨 Experiment 复用。Gate 正是重新登记的对象，计算只需毫秒；可视化要画出当前 Gate 的判定。

### 3. 查找时现场计算来源的键，不写索引

查找在 `LocalNotebookPlatform` 中进行：遍历 `runs/*/config.json`，对每个 Attempt（从新到旧）读取同名 Stage 记录。若状态为 `completed`，用来源的 `config.json` 与该记录的 `input_artifacts` 重新计算复用键；键相同且全部输出 checksum 通过时返回。

这样旧的 `runs/` 无需迁移就能被复用，也没有需要加锁的共享可变文件，与 `add-local-experiment-queue` 对 registry 并发的顾虑不冲突。代价是每个 Stage 扫描一遍 `runs/`：本地规模为几十个 Experiment，只读 JSON，毫秒级。跨节点或大规模时应改为数据库索引，属于服务器扩展路线，不在本 change 内。

查找顺序：先用恢复源（现有 `_reusable`，语义不变），再跨 Experiment 查找。同一 Experiment 的其他 Attempt 也会被跨 Experiment 查找覆盖，但恢复源优先，这样恢复记录保持原样。

### 4. 复用来的 Artifact 按引用读取

跨 Experiment 复用后，科学评估结果、训练历史等 Artifact 的清单位于来源 Experiment 的 `artifacts/`。当前 run 的 `LocalArtifactRepository` 找不到它们，而 Gate 和可视化目前按 artifact id 在本 Experiment 清单中查找。

改为在 run 内维护 `artifact_id → ArtifactRef` 映射，包括本 Attempt 的产物与复用得到的 Stage 输出。按引用读取时先校验 checksum 再读文件。Artifact 不复制、不重新登记，来源 Experiment 仍是唯一生产者。

### 5. 记录

跨 Experiment 复用的 Stage 元数据包含：

```json
{"reused_from": "<attempt-id>", "reused_from_experiment": "<experiment-id>", "reuse_key": "sha256:..."}
```

同 Experiment 恢复复用只写 `reused_from`，与现有记录一致。日志行为 `reused verified model from <experiment>/<attempt>`。

### 6. Notebook

`CONFIG["accuracy_gate"]` 改为 `comparison_rule = "paired-relative-non-inferiority"`、`tolerance = 0.15`。22 个点的 Experiment ID 全部变化，每个点新建 `attempt-0001`：数据集按 DatasetKey 复用；训练、科学评估，以及 Gate PASS 时的性能评估，跨 Experiment 复用 0.02 下的 Experiment；Gate 与可视化重新计算。Markdown 如实说明容差是在看到 0.02 的结果后重新登记的，并给出理由。

## Risks / Trade-offs

- **代码变化不使复用失效。** 修复训练代码后，复用键不变，会复用旧模型。这与现有同 Experiment 恢复的语义相同。需要重训时应修改 `model.architecture_version` 等配置字段，或删除相应 `runs/`。design 记录这一点，不在本 change 内解决。
- **删除表写错会导致错误复用。** 缓解方式：删除表只有 3 行，每一项都有测试覆盖，包括只改 Gate 时复用、改训练字段时不复用、改实验名时复用。
- **事后改容差。** 这是在看到结果后作出的重新登记。缓解方式：旧 Experiment 与其判定保留不动，新 Experiment 的配置快照记录新规则，Notebook 说明原因；相对容差在低 p 更严格（L=6、p=0.05 从 PASS 变为 FAIL），并非只放宽判定。
- **扫描成本随 `runs/` 线性增长。** 本地可接受；规模化时再换索引。
