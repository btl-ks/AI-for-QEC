# Proposal

## Why

项目当前有 7 条随机流，全部以人工挑选的裸偏移量从主种子派生，偏移量分散在 5 个文件里。其中已存在一处实际碰撞：RBM 训练器的 DataLoader 洗牌流与对比散度（CD）采样流都使用 `first_seed(config) + 41`，两条本应独立的流从同一状态出发。另有一处同值派生：模型初始化与训练集生成都使用 `base + 0`，仅因分属 torch 与 numpy 两套生成器才没有产生相同序列。

这类缺陷不改变任何可观察的输入输出行为——程序不崩溃、结果可复现、损失正常下降、行覆盖完整——因此现有的可复现性单元测试、管道冒烟测试、mypy 与 ruff 全部无法发现。可复现性与独立性是两条不同的性质，项目此前只测了前者。

本变更是一个可实施的 child change，不是 umbrella baseline。

## What Changes

- 新增 `derive_seed(base, stream)`：按流名做 SHA-256 派生，取代裸偏移量。新增一条流只需取一个名字，不必人工挑选不冲突的数字。
- 将全部 7 条具名随机流改为经由该函数派生：模型初始化、train/validation/test 三条数据生成流、DataLoader 洗牌、CD 采样、评估。
- 消除 `ai_qec/data/generators/qec_generator.py` 中与 `SPLIT_SEED_OFFSETS` 内容相同的重复字面量常量表，改为引用已导出的单一来源。
- 新增回归测试：具名流两两不同、同输入派生确定、结果落在 int63 范围内。
- 按项目硬约束在 `ai_qec/notebook_api.py` 导出新公开 API。
- **BREAKING**：生成、训练、评估三端的随机流全部变化。已生成数据集的内容与其 `dataset_id` 不再对应，历史 run 的结果无法用新代码重现。本变更不提供兼容模式或迁移路径；用户已确认接受该代价。

## Capabilities

### New Capabilities

- `research/experiment-lifecycle`：补充具名随机流派生的可观察要求。该能力路径已由 umbrella change `establish-qec-research-platform-requirements` 声明，但主规格尚未同步（`openspec/specs/` 为空），因此本变更以 ADDED Requirement 形式追加，不与 umbrella 的既有 Requirement 重叠。

### Modified Capabilities

无。主规格当前为空，没有可修改的既有 Requirement。

## Impact

- **代码**：`ai_qec/utils/config.py`（新增派生函数）、`ai_qec/models/registry.py`、`ai_qec/data/generators/qec_generator.py`、`ai_qec/data/generators/toric_generator.py`、`ai_qec/training/trainers/rbm.py`、`ai_qec/training/evaluation/toric_rbm.py`、`ai_qec/notebook_api.py`。
- **测试**：`tests/unit/test_reproducibility.py` 新增具名流回归测试；既有的训练可复现性测试应继续通过，因为本变更只改变流之间的独立性，不破坏同输入同输出。
- **数据与产物**：所有已生成数据集作废；`runs/` 下历史 run 的数值结果不可用新代码重现。二者均不入 Git，不产生仓库层面的迁移工作。
- **依赖**：无新增依赖，仅使用标准库 `hashlib`。
- **Gate 依赖**：本变更独立于 P0.14 与 P0.19，可并行推进；它不改变 dataset 身份的计算方式，只改变同一身份下生成的内容。
