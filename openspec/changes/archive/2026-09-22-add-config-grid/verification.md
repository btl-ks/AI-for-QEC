# Verification

## Scope

本验证证明：`qec.config_grid` / `qec.with_overrides` / `qec.GridPoint` 按声明展开配置网格并在返回任何配置前拒绝非法声明；论文 Notebook 改用两个 `config_grid` 声明后，paper profile 端到端生成 22 个新 Experiment，数据集全部复用。它不证明 runtime、训练或评估行为有任何变化（本 change 未改动这些行为），也不证明 Accuracy Gate 的容差适合高错误率点。

环境：WSL2 Linux，NVIDIA GeForce RTX 4060 Laptop GPU，conda 环境 `quantum`（Python 3.12.14）；无运行时依赖的基础环境为 Python 3.13.12。

## Evidence

### 单元与集成测试

命令：

```bash
/home/zephy/miniconda3/envs/quantum/bin/python -m unittest discover -s tests
python3 -m unittest discover -s tests
python3 -m compileall -q ai_qec tests
```

结果：`quantum` 环境 79 个测试全部通过（4.8 s）；基础环境 79 个测试通过，其中 28 个需要运行时依赖的测试被跳过；compileall 通过。关键行为证据：

- `test_config_grid.test_cartesian_product_with_coupled_and_shared_overrides`：笛卡尔积顺序、耦合覆盖与共享覆盖生效，基础配置不被修改。
- `test_config_grid.test_invalid_declarations_fail_before_any_point_is_returned`：拼错路径、同一字段重复覆盖、耦合覆盖缺少轴取值、实验名重复都在返回任何配置前失败。
- `test_config_grid.test_with_overrides_reports_the_full_missing_path`：错误信息包含完整缺失路径。
- `test_paper_notebook.test_paper_grid_matches_figure_3`：paper 网格为 22 个点，隐藏单元数与 epoch 数按 L 耦合。

### 论文 Notebook（paper profile）

命令（`quantum` 环境，cwd 为 `paper/srcs`，nbclient 执行全部 cells）：

```python
NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "paper/srcs"}}).execute()
```

执行过程如实记录：

1. 首次执行于 2026-09-22 04:12 开始。04:37 前完成 15 个点；之后主机休眠、WSL 暂停，12:14 恢复后继续。12:21:17 VS Code 以 `renderer disconnected for too long` 终止旧窗口的扩展宿主，启动该执行的 Claude 会话随之结束，其后台进程树被杀，内核在 12:21:19 死亡（`DeadKernelError`）。此时 18 个点已完成，`l6-p0.12` 的 `attempt-0001` 停在 `running`，最新恢复 checkpoint 为 epoch 14。dmesg 无 OOM 或段错误记录。
2. 重新执行（12:37 开始，362 s，8 个 cells 均无错误输出）：
   - 18 个已完成的点各创建 `attempt-0002`，数据集、训练、科学评估与性能结果均复用 `attempt-0001`。
   - `l6-p0.12`：`attempt-0001` 因 owner 进程已不存在被标记为 `interrupted`；`attempt-0002` 从 `attempt-0001.recovery-epoch-0014` 继续，训练 26 个 epoch 后完成（`resumed_from.epoch = 14`，`epochs_trained_in_attempt = 26`）。
   - `l6-p0.13`、`l6-p0.14`、`l6-p0.15` 为首次训练。
3. 数据集：两次执行共 41 次 dataset Stage 全部为 `reused verified`，没有生成新数据集（DatasetKey 不含实验名）。
4. 用当前代码离线计算 22 个点的 Experiment ID，与 `runs/` 中的目录逐一吻合（`torlai-melko-2017-paper-l{L}-p{p:.2f}-<digest>`）；旧格式 `…-l4-p008` 目录保留未删除。

最终结果（10 000 test shots，Wilson 95% 区间，Gate 为配对非劣效、绝对容差 +0.02）：

```text
 L     p |               RBM P_fail timeouts |              MWPM P_fail | gate
 4  0.05 |  0.0754 [0.0704, 0.0807]        0 |  0.0763 [0.0713, 0.0817] | PASS
 4  0.06 |  0.1167 [0.1106, 0.1231]        1 |  0.1168 [0.1107, 0.1232] | PASS
 4  0.07 |  0.1597 [0.1527, 0.1670]        0 |  0.1561 [0.1491, 0.1633] | PASS
 4  0.08 |  0.2044 [0.1966, 0.2124]        0 |  0.1960 [0.1883, 0.2039] | PASS
 4  0.09 |  0.2450 [0.2367, 0.2535]        0 |  0.2366 [0.2284, 0.2450] | PASS
 4  0.10 |  0.2953 [0.2864, 0.3043]        0 |  0.2820 [0.2733, 0.2909] | FAIL
 4  0.11 |  0.3414 [0.3322, 0.3508]        0 |  0.3314 [0.3222, 0.3407] | PASS
 4  0.12 |  0.4005 [0.3909, 0.4101]        0 |  0.3773 [0.3678, 0.3868] | FAIL
 4  0.13 |  0.4406 [0.4309, 0.4504]        0 |  0.4121 [0.4025, 0.4218] | FAIL
 4  0.14 |  0.4833 [0.4735, 0.4931]        0 |  0.4584 [0.4487, 0.4682] | FAIL
 4  0.15 |  0.5266 [0.5168, 0.5364]        0 |  0.5022 [0.4924, 0.5120] | FAIL
 6  0.05 |  0.0431 [0.0393, 0.0473]       88 |  0.0384 [0.0348, 0.0423] | PASS
 6  0.06 |  0.0712 [0.0663, 0.0764]       84 |  0.0715 [0.0666, 0.0767] | PASS
 6  0.07 |  0.1077 [0.1018, 0.1139]       76 |  0.1076 [0.1017, 0.1138] | PASS
 6  0.08 |  0.1611 [0.1540, 0.1684]       66 |  0.1611 [0.1540, 0.1684] | PASS
 6  0.09 |  0.2151 [0.2072, 0.2233]       59 |  0.2146 [0.2067, 0.2228] | PASS
 6  0.10 |  0.2759 [0.2672, 0.2847]       75 |  0.2731 [0.2645, 0.2819] | PASS
 6  0.11 |  0.3401 [0.3309, 0.3494]       81 |  0.3363 [0.3271, 0.3456] | PASS
 6  0.12 |  0.4048 [0.3952, 0.4145]      135 |  0.3891 [0.3796, 0.3987] | FAIL
 6  0.13 |  0.4708 [0.4610, 0.4806]      299 |  0.4437 [0.4340, 0.4535] | FAIL
 6  0.14 |  0.5346 [0.5248, 0.5444]      407 |  0.5057 [0.4959, 0.5155] | FAIL
 6  0.15 |  0.5760 [0.5663, 0.5857]      876 |  0.5435 [0.5337, 0.5532] | FAIL
```

与 `reproduce-torlai-melko-2017` 的 verification 表（旧格式实验名、一次性训练）逐行 `diff`，22 行完全相同。本 change 不改变训练或评估，这是预期结果。

硬中断恢复的对照：旧格式 `l6-p012-eb59a5357ef9/attempt-0001` 在同一配置下一次性训练 40 个 epoch，没有中断；新 `l6-p0.12-c8047f2dc98a/attempt-0002` 在进程被杀后从 epoch 14 恢复。两者的 `final.pt` 在 CUDA 上 `model_identity` 相同，`weight (128, 108)`、`visible_bias (108,)`、`hidden_bias (128,)` 均 `torch.equal` 逐位相同。

Gate 在高错误率下 FAIL 属于科学结果，不是本 change 的缺陷；容差调整由后续 change 处理。

### OpenSpec

```bash
openspec validate --all --strict --no-interactive
```

结果：11 项全部通过。

## Gaps

- 未测试 smoke profile 的端到端执行（其展开由单元测试覆盖）。
