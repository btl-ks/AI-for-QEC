# Verification

## Scope

证明 Notebook setup cell 缩减为两句后，定义 cells 与完整 Run All 行为不变，且 `find_project_root` 的定位与失败行为符合 design。不涉及 runtime 或实验结果的变化。

## Evidence

### 内核环境

命令：`/home/zephy/miniconda3/envs/quantum/bin/python -m pip install --no-deps -e .`

结果：安装 `ai-for-qec 0.1.0.dev0`（editable）；在 `/tmp` 中 `import ai_qec.notebook_api` 解析到 `/home/zephy/workspace/QEC/ai_qec/notebook_api.py`。

### 测试

命令：

```bash
/home/zephy/miniconda3/envs/quantum/bin/python -m unittest discover -s tests
python3 -m unittest discover -s tests
```

结果：`quantum` 环境 76 个测试全部通过；基础环境全部通过（28 个需要运行时依赖的测试跳过）。`tests/unit/test_paths.py` 覆盖：从 `paper/srcs` 与 cwd 定位到仓库根目录；从仓库外目录退回到 editable 源码目录；模拟 site-packages 安装且 cwd 不在仓库内时抛出 `FileNotFoundError`。`tests/unit/test_paper_notebook.py` 验证新的定义 cells 可执行且不创建 runtime。

### Notebook

命令：nbclient 在 `quantum` 内核中执行 `paper/srcs/ai_for_qec_workflow.ipynb`（cwd 为 `paper/srcs`，setup cell 不再修改 `sys.path`）。

结果：9 s 完成，无错误输出；5 张图（图 3、图 4、3 张单点图）；22 个点全部复用已校验模型，0 个新数据集。

### OpenSpec

`openspec validate simplify-notebook-setup --strict --no-interactive` 与 `openspec validate --all --strict --no-interactive` 在归档前通过。
