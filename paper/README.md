# Paper notebooks

`srcs/` 保存论文实验的 `.ipynb` 源码；Notebook 是实验配置、阶段编排、可视化入口和研究解释，不承载数据生成、训练、解码、Artifact 或恢复实现。`pdf/` 保存论文原文。

## ai_for_qec_workflow.ipynb

复现 G. Torlai and R. G. Melko, *Neural Decoder for Topological Codes*, PRL 119, 030501 (2017)：

- `run_paper(runtime, config)` 保持标准顺序：Start/Recover → Resolve Dataset → Train → Scientific Evaluation → Accuracy Gate → 仅 PASS 时 Performance → Visualize → Finish。
- `paper` profile 覆盖论文图 3 的网格（L ∈ {4, 6}，p = 0.05…0.15，共 22 个独立 Experiment），并绘制图 3 与图 4；`smoke` profile 只跑 3 个小点。
- 定义 cells 无副作用；只有带 `run-experiment` 标签的 cells 会构造 `qec.LocalNotebookPlatform` 并执行实验。
- 运行产物写入仓库根目录下的 `datasets/` 与 `runs/`（git 忽略）；再次 Run All 会创建新 Attempt 并复用已校验的数据集、模型与评估。

内核环境需要以 editable 方式安装本项目及运行时依赖：在仓库根目录执行 `pip install -e ".[runtime]"`（本机 conda 环境 `quantum` 已安装）。之后在该内核中打开 Notebook 并 Run All 即可；`qec.find_project_root()` 会从 Notebook 所在目录向上定位仓库根目录。源码中不保存 cell 输出。

`torlai_melko_2017.ipynb` 是旧实现的参考 Notebook，依赖本仓库不存在的旧 API，不能直接运行。
