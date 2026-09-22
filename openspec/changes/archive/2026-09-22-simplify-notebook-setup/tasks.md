# Tasks

## 1. 实现

- [x] 1.1 新增 `ai_qec/utils/paths.py` 的 `find_project_root` 并从 `ai_qec.notebook_api` 导出，以 `tests/unit/test_paths.py` 验证子目录定位、editable 退回与找不到时的 `FileNotFoundError`
- [x] 1.2 把 Notebook setup cell 缩减为两句，并把 `copy`、`Mapping` 移到使用它们的 cell，以 `tests/unit/test_paper_notebook.py` 验证定义 cells 仍可执行且不创建 runtime
- [x] 1.3 更新 `paper/README.md` 的内核环境说明

## 2. 验证与归档

- [x] 2.1 在 `quantum` 与基础环境运行全部测试，在 `quantum` 内核中用 nbclient（cwd 为 `paper/srcs`）执行 Notebook 并确认无错误、结果复用，把证据写入 `verification.md`
- [x] 2.2 运行 `openspec validate --all --strict --no-interactive` 后归档
