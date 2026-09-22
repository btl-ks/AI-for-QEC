# Tasks

## 1. 相对容差 Gate

- [x] 1.1 `paired_difference_interval` 增加 `scale` 参数（MOVER 线性对比），以 `tests/unit/test_statistics.py` 验证 scale=1 与原实现逐位一致、scale>1 区间随之下移，以及边界上 Monte Carlo 的 PASS 频率落在 1.5%–3.5%
- [x] 1.2 新增 `paired-relative-non-inferiority` 规则（`ai_qec/evaluation/scientific/local.py`、`ai_qec/experiment/config.py`），以 `tests/unit/test_pipeline_and_gate.py` 验证低 p 算例 FAIL、高 p 算例 PASS、证据文件含 δ 与比值，以 `tests/integration/test_local_runtime.py` 验证负 δ 在创建 `runs/`、`datasets/` 前失败

## 2. 跨 Experiment Stage 复用

- [x] 2.1 新增 Stage 复用键计算与删除表，以单元测试验证每个删除字段不改变对应键、其余字段与输入引用改变键
- [x] 2.2 `LocalNotebookRun` 在恢复源不可复用时跨 Experiment 查找训练、科学评估与性能评估 Stage，按引用读取复用来的 Artifact，并记录来源 Experiment、Attempt 与复用键；以 `tests/integration/test_local_runtime.py` 验证只改 Gate 时不训练且 Gate 重新计算、改 `training.epochs` 时重新训练、只改实验名时复用、来源模型被篡改时重新训练

## 3. Notebook 与验证

- [x] 3.1 Notebook 的 Gate 改为相对 15% 并更新说明，以 `tests/unit/test_paper_notebook.py` 验证配置
- [x] 3.2 在 `quantum` 与基础环境运行全部测试；在 `quantum` 内核中执行 Notebook（paper profile），确认 22 个新 Experiment 完成、训练 epoch 总数为 0、Gate 判定与预览一致，把结果写入 `verification.md`
- [x] 3.3 更新 `openspec/capabilities.yaml` 证据，运行 `openspec validate --all --strict --no-interactive` 后归档
