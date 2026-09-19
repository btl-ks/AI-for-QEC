# Design

## Context

见 [proposal.md](proposal.md) 的 Why。补充设计所需的现状：项目同时使用 numpy 与 torch 两套随机数生成器，7 条流分布在 5 个文件中，派生方式为 `base + <人工挑选的偏移量>`。偏移量取值现为 `0`、`41`、`10_000`、`20_000`、`1_000_000 + index`，其中 `41` 被两条流同时使用，`0` 被模型初始化与训练集生成同时使用。

约束：主种子来自配置 `reproducibility.seeds[0]`，本变更不改变它的读取方式，也不改变 `dataset_id` 的计算方式。

## Goals / Non-Goals

**Goals**
- 使新增一条随机流只需取一个未使用的名字，不需要人工确认数字不冲突。
- 使流名拼写错误在测试阶段失败，而不是静默产生一条新流。
- 让 numpy 与 torch 两侧使用同一套派生结果。

**Non-Goals**
- 不改变主种子的配置来源与多 seed 语义（属 P0.8，已完成）。
- 不改变 `dataset_id` 的计算方式（属 P0.14）。
- 不为已生成数据集提供迁移或兼容模式。

## Decisions

### 1. 派生引擎使用 `numpy.random.SeedSequence`，不自建哈希混合

**备选方案**：直接取 `hashlib.sha256(f"{base}:{name}")` 的前 64 位作为种子。

**选择 SeedSequence 的理由**：项目规则要求优先成熟库。SeedSequence 是 NumPy 专为「从单一熵源派生多条互不相关的流」设计的组件，其输出状态经过混合以避免子流之间的相关性。自建哈希能保证「两个种子值不同」，但这不等于「两条流不相关」——而后者才是本变更要的性质。SeedSequence 对此有明确设计目标与统计验证。

sha256 仍然使用，但只用于把**流名**转换为稳定整数作为 entropy 的一部分：Python 内置 `hash()` 按进程加盐，跨进程不稳定，不能用于需要可复现的身份。

派生形式为 `SeedSequence([base, sha256_int(name)]).generate_state(1, dtype=uint64)`，右移一位落入 int63，兼容 torch 与 numpy 的播种接口。

### 2. 独立模块承载随机流职责

新增 `ai_qec/utils/random_streams.py`，而非并入 `ai_qec/utils/config.py`。

**备选方案**：放在 `first_seed` 旁边（同一文件）。

**否决理由**：AGENTS.md 约束 5 要求单一职责。`config.py` 的职责是配置解析与严格校验，让它同时承担随机数派生会把两类关注点耦合在一个已经很大的模块里。`first_seed` 保留在 `config.py`（它确实是在读配置），新模块只负责从一个已取得的主种子派生。

### 3. 显式流名注册表

模块导出 `STREAM_NAMES: frozenset[str]`，枚举全部具名流；`derive_seed` 对未注册的名字 fail loudly。

**备选方案**：接受任意字符串。

**否决理由**：把 `"cd"` 误写成 `"dc"` 会静默产生第 8 条流并改变结果，与本变更要消除的缺陷同类。注册表同时让「新增一条具名流」这一 Scenario 可测。

### 4. 返回整数种子，不返回构造好的生成器

调用方自行用返回值构造 `torch.Generator` 或 `np.random.default_rng`。

**备选方案**：返回已构造的生成器。

**否决理由**：两套库的生成器类型与设备语义不同，返回生成器需要额外的 device 参数或两个函数，会把库选择耦合进派生层。

### 5. 不提供兼容模式

派生方式变更后不保留旧路径。用户已确认接受已生成数据集作废（见 proposal.md 的 BREAKING 条目）。保留 legacy 分支会长期存在且无人验证。

## Risks / Trade-offs

- **已生成数据集与历史 run 结果全部作废** → 数据集、runs 与 checkpoint 均不入 Git，重新生成即可；在 `docs/OPTIMIZATION_PLAN.md` 记录该断点时间与原因，便于日后解释历史结果为何不可重现。
- **流名拼写错误** → 注册表 + fail loudly；测试断言注册表内名字两两派生不同。
- **未来 NumPy 改变 SeedSequence 的 entropy 语义会使派生结果变化** → 接受该风险。项目已用 `uv.lock` 锁定 numpy 版本，且 `env_id` 层面的版本记录已由 `package_versions()` 覆盖；不额外引入派生算法版本号，避免增加一个需要人工维护的数字。
- **既有训练可复现性测试是否会失败** → 不会。该测试断言「同配置两次执行结果相同」，与流的绝对取值无关，只与派生是否确定有关。

## Migration Plan

1. 新增 `ai_qec/utils/random_streams.py` 与注册表，并在 `ai_qec/notebook_api.py` 导出。
2. 逐个替换 7 处派生点：模型初始化、train/validation/test 生成、洗牌、CD 采样、评估。
3. 删除 `qec_generator.py` 中与 `SPLIT_SEED_OFFSETS` 重复的字面量常量表，改为引用单一来源。
4. 运行回归测试与冒烟；重新生成 smoke 数据集。
5. 在 `docs/OPTIMIZATION_PLAN.md` 记录 P4.18 与随机流断点。

**回滚**：单次 `git revert` 即可。本变更不产生持久化状态迁移，回滚后需再次重新生成数据集。

## 需求到证据追踪

| Requirement / Scenario | 功能设计 | 实现任务 | 测试证据 |
|---|---|---|---|
| 具名随机流相互独立且可确定性重建 / 同一主种子下的两条不同具名流 | 决策 1、3 | 任务 1、2 | `test_registered_streams_are_pairwise_distinct` |
| 同上 / 重复重建同一具名流 | 决策 1 | 任务 1 | `test_derivation_is_deterministic` |
| 同上 / 主种子改变 | 决策 1 | 任务 1 | `test_base_seed_change_moves_every_stream` |
| 同上 / 新增一条具名流 | 决策 3 | 任务 1、2 | `test_unregistered_stream_name_fails_loudly` |
| 全部 Scenario 的端到端体现 | 决策 2、4、5 | 任务 3、4、5 | `pytest tests/`、`scripts/run_experiment.py --config configs/experiment.smoke.yaml` |
