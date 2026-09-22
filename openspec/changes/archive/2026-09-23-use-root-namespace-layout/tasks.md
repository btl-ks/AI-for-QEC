# Tasks

## 1. 根目录打包布局

- [x] 1.1 将 setuptools discovery 改为根目录 `where = ["."]`、`include = ["ai_qec*"]` 和 namespace 模式，并用 `uv build --wheel --no-build-isolation` 验证配置
- [x] 1.2 将 `src/ai_qec` 迁回根目录 `ai_qec`、清理生成的空 `src` 目录，并用 `test -d ai_qec`、`test ! -d src/ai_qec` 验证唯一源码树
- [x] 1.3 保持源码与测试树零初始化文件，并用 `find ai_qec tests -type f -name __init__.py` 验证结果为空

## 2. 导入与行为验证

- [x] 2.1 更新 packaging contract 测试以覆盖根目录直导入和受限 discovery，并用 `python -m unittest discover -s tests/unit -p test_packaging.py -v` 验证
- [x] 2.2 运行 `python -m compileall -q ai_qec tests`、unit 与 integration 测试，验证公共 facade、项目根定位和领域行为未回归
- [x] 2.3 在仓库外隔离环境安装 wheel，验证 64 个 `ai_qec` 模块、零 `__init__.py`、零误收录 package、130 个公共导出和可选 runtime 零加载

## 3. 文档与治理

- [x] 3.1 更新 README、架构背景和 capability evidence 的当前物理路径，并用 `rg 'src/ai_qec' README.md openspec/capabilities.yaml docs/AI_for_QEC_Codex_Architecture_Context.md` 审核结果为空
- [x] 3.2 同步锁文件、验证 capability schema，并运行 `openspec validate --all --strict --no-interactive`
- [x] 3.3 写入真实命令与结果到 `verification.md`、完成任务清单，并用 `git diff --check` 与 change strict validation 完成归档前检查
