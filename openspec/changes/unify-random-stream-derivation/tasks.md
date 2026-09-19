# Tasks

父任务：`docs/OPTIMIZATION_PLAN.md` 的 P4.18（随本变更新增）。
需求来源：`specs/research/experiment-lifecycle/spec.md` 的 Requirement「具名随机流相互独立且可确定性重建」。

## 1. 派生层

- [ ] 1.1 新增 `ai_qec/utils/random_streams.py`，实现基于 `numpy.random.SeedSequence` 的 `derive_seed(base, stream)` 与 `STREAM_NAMES` 注册表（design 决策 1、2、3、4）；未注册的流名 fail loudly。
      验证：`.venv/bin/python -m pytest tests/unit/test_reproducibility.py -q -k "stream"`，其中 `test_registered_streams_are_pairwise_distinct`、`test_derivation_is_deterministic`、`test_base_seed_change_moves_every_stream`、`test_unregistered_stream_name_fails_loudly` 分别覆盖四个 Scenario。
- [ ] 1.2 在 `ai_qec/notebook_api.py` 导入 `derive_seed` 与 `STREAM_NAMES` 并加入 `__all__`（AGENTS.md 硬约束 4）。
      验证：`.venv/bin/python -c "from ai_qec.notebook_api import derive_seed, STREAM_NAMES"` 退出码为 0。

## 2. 替换既有派生点

- [ ] 2.1 训练侧三条流改用注册名：模型初始化（`ai_qec/models/registry.py`）、DataLoader 洗牌与 CD 采样（`ai_qec/training/trainers/rbm.py`）。消除 `first_seed(config) + 41` 的重复使用（Scenario：同一主种子下的两条不同具名流）。
      验证：`grep -rn "+ 41" ai_qec/training/trainers/rbm.py` 无输出；`.venv/bin/python -m pytest tests/unit/test_rbm_data_training.py -q` 通过。
- [ ] 2.2 生成侧三条流改用注册名（`ai_qec/data/generators/toric_generator.py`、`ai_qec/data/generators/qec_generator.py`），并删除 `qec_generator.py` 中与 `SPLIT_SEED_OFFSETS` 重复的字面量常量表，改为引用单一来源（design Migration Plan 第 3 步）。
      验证：`grep -n "10_000" ai_qec/data/generators/qec_generator.py` 无输出；`.venv/bin/python -m pytest tests/unit/test_reproducibility.py -q` 通过。
- [ ] 2.3 评估流改用注册名并按样本索引派生（`ai_qec/training/evaluation/toric_rbm.py`），消除 `+ 1_000_000 + index`。
      验证：`grep -n "1_000_000" ai_qec/training/evaluation/toric_rbm.py` 无输出；`.venv/bin/python -m pytest tests/unit/test_toric_rbm.py -q` 结果不劣于变更前基线。
- [ ] 2.4 全仓库确认不再有裸偏移式派生。
      验证：`grep -rnE "first_seed\(config\) \+|base_seed \+" ai_qec/ --include="*.py"` 无输出。

## 3. 收口与证据

- [ ] 3.1 运行全量测试与冒烟，结果不劣于变更前基线（基线：1 failed / 57 passed / 5 skipped，failure 为 `.venv` 下 `execution.conda_env` 校验，与本变更无关）。
      验证：`.venv/bin/python -m pytest tests/ -q`；`.venv/bin/python scripts/run_experiment.py --config configs/experiment.smoke.yaml` 返回 success。
- [ ] 3.2 lint 与类型检查不劣于基线（基线：ruff 13、mypy 2）。
      验证：`.venv/bin/python -m ruff check ai_qec/ tests/`；`.venv/bin/python -m mypy ai_qec/utils/random_streams.py`。
- [ ] 3.3 在 `docs/OPTIMIZATION_PLAN.md` 新增并勾选 P4.18，记录随机流断点的时间与原因（已生成数据集与历史 run 结果不可用新代码重现）。
      验证：`grep -n "P4.18" docs/OPTIMIZATION_PLAN.md` 有输出且标记为已完成。
- [ ] 3.4 运行 OpenSpec verify 工作流并按 `openspec/templates/verification.md` 生成 `verification.md`，记录受测 commit、环境、需求/场景/测试矩阵、命令与结果、证据路径与哈希、限制与结论。
      验证：`openspec validate unify-random-stream-derivation --strict` 通过，且 `verification.md` 存在、无 CRITICAL 未解决项。
